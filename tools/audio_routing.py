"""Linux host-API names and channel-selector helpers for PortAudio routing."""

JACK_API = "JACK Audio Connection Kit"
ALSA_API = "ALSA"
HOSTAPI_LABELS = {JACK_API: "JACK", ALSA_API: "ALSA"}
# PipeWire's shared ALSA PCMs.  Raw hw:* devices are held open by PipeWire.
ALSA_SHARED_PCMS = ("pipewire", "default", "pulse")


def is_native_api(api_name):
    """JACK owns its period size, so its streams must use blocksize=0."""
    return api_name == JACK_API


def select_channels(indata, selectors):
    """Pick the selected input columns; no selectors keeps every column."""
    if not selectors:
        return indata
    return indata[:, selectors]


def scatter_mono(outdata, mono, selectors):
    """Write a mono block to the selected output columns (or all of them)."""
    sample_count = mono.shape[0]
    if selectors:
        outdata[:sample_count, selectors] = mono[:, None]
    else:
        outdata[:sample_count, :] = mono[:, None]

