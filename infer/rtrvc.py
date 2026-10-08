import traceback
from time import time as ttime
import faiss
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from engine.assets import RMVPE_PATH
from infer.hubert import extract_hubert_features, load_hubert_model
from tools.cuda_graph import run_cuda_graph


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


#: RMVPE pitch context (seconds).  tools/pitch_bench.py on real speech: the
#: old block + 50 ms window (0.32 s at 0.25 s blocks) had 2.7% gross pitch
#: errors and octave jumps; 1.28 s cut gross errors by 60%, removed octave
#: errors and halved voicing mistakes for about 1 ms per block.  Returns
#: flatten beyond that.
RMVPE_CONTEXT_SECONDS = 1.28
#: RMVPE windows are whole multiples of 32 mel frames (5120 samples) minus a hop.
RMVPE_WINDOW_STEP = 5120


def rmvpe_window(block_frame_16k, available):
    """Samples of 16 kHz context to give RMVPE, at least the newest block.

    The window must end at the newest sample and fit in the ``available``
    context; it is rounded to RMVPE's frame structure.
    """
    wanted = max(block_frame_16k + 800, int(RMVPE_CONTEXT_SECONDS * 16000))
    window = RMVPE_WINDOW_STEP * ((wanted - 1) // RMVPE_WINDOW_STEP + 1) - 160
    largest = RMVPE_WINDOW_STEP * ((available + 160) // RMVPE_WINDOW_STEP) - 160
    minimum = RMVPE_WINDOW_STEP * ((block_frame_16k + 800 - 1) // RMVPE_WINDOW_STEP + 1) - 160
    return max(min(window, largest), min(minimum, available))


#: Longest unvoiced run (10 ms frames) treated as a pitch-detector dropout and
#: bridged; longer runs stay unvoiced so the synthesiser renders them as noise.
#: tools/consonant_bench.py on real speech (streaming RMVPE): 3-5 frames cut
#: forced pitch on consonants/breaths most (28% -> 18%) but left 1.6-2% of
#: vowel frames unvoiced (audible breaks); 8 frames cut it to 22% with voiced
#: frames unchanged within the noise floor.
MAX_F0_GAP_FRAMES = 8
#: Restore the old behaviour (every frame voiced); for benchmarks only.
FULL_F0_INTERPOLATION = False


def fill_short_gaps(f0, max_gap=None):
    """Bridge brief unvoiced dropouts inside voiced speech; keep the rest unvoiced.

    Runs of at most ``max_gap`` zero frames with voiced frames on both sides
    are filled by interpolating in log-frequency.  Longer runs, and runs that
    touch either end of the analysis window, stay 0 so RVC's NSF source uses
    noise for them (breaths, s/sh/f/t/h) instead of a forced pitch.
    """
    if max_gap is None:
        max_gap = MAX_F0_GAP_FRAMES
    f0 = np.array(f0, dtype=np.float64)  # a copy; the caller's array is untouched
    voiced = f0 > 0
    if FULL_F0_INTERPOLATION:
        if np.any(voiced):
            f0[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], f0[voiced])
        return f0
    index, length = 0, f0.shape[0]
    while index < length:
        if voiced[index]:
            index += 1
            continue
        end = index
        while end < length and not voiced[end]:
            end += 1
        if index > 0 and end < length and end - index <= max_gap:
            left, right = np.log(f0[index - 1]), np.log(f0[end])
            steps = np.arange(1, end - index + 1) / (end - index + 1)
            f0[index:end] = np.exp(left + (right - left) * steps)
        index = end
    return f0


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
    ) :
        """
        初始化
        """
        try:
            # global config
            self.config = config
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.formant_shift = formant
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.is_half = config.is_half
            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
            self.pth_path = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.infer_count = 0

            self.resample_kernel = {}

            if last_rvc is None:
                self.model = load_hubert_model(self.device, self.is_half)
            else:
                self.model = last_rvc.model

            self.net_g = None

            def set_synthesizer():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.model_fcpe = last_rvc.model_fcpe
        except Exception:
            # Re-raise: a half-initialised RVC (e.g. missing tgt_sr) only
            # produces a confusing error later on.
            printt(traceback.format_exc())
            raise

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("Index search enabled")
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        f0 = fill_short_gaps(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt("Loading RMVPE model")
            self.model_rmvpe = RMVPE(
                RMVPE_PATH,
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 = fill_short_gaps(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        f0 = fill_short_gaps(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            feats = extract_hubert_features(
                self.model,
                feats,
                self.version,
                padding_mask=padding_mask,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                # IVF indexes created with nprobe=1 can return -1 when the
                # single probed list is empty. That is a sparse search miss,
                # not a corrupt or unsupported index file. Retry wider and
                # retain the wider setting for following realtime blocks.
                if (ix < 0).any() and hasattr(self.index, "nprobe"):
                    nlist = int(getattr(self.index, "nlist", 0))
                    current_nprobe = max(1, int(self.index.nprobe))
                    retry_nprobe = min(max(current_nprobe * 4, 4), nlist or 4)
                    if retry_nprobe > current_nprobe:
                        self.index.nprobe = retry_nprobe
                        score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        "Invalid index: use added_xxxx.index, not trained_xxxx.index"
                    )
            else:
                if report_status:
                    printt("Index search failed or is disabled")
        except Exception:
            traceback.print_exc()
            printt("Index search failed")
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                # RMVPE gains a lot from past context; FCPE and PM do not use it.
                f0_extractor_frame = rmvpe_window(block_frame_16k, input_wav.shape[0])
            pitch, pitchf = self.get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                f0method,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        t5 = ttime()
        if report_status:
            printt(
                "Timing: features=%.3fs, index=%.3fs, pitch=%.3fs, model=%.3fs",
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
