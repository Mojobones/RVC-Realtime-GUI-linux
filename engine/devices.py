"""PortAudio device catalog for JACK (PipeWire) and PipeWire's ALSA PCMs."""

from dataclasses import dataclass, field

from tools.audio_routing import ALSA_API, ALSA_SHARED_PCMS, HOSTAPI_LABELS, JACK_API


@dataclass
class Endpoint:
    label: str
    index: int
    api: str
    name: str
    selectors: list = field(default_factory=list)


def channel_pairs(channel_count):
    pairs = []
    for first in range(0, int(channel_count), 2):
        second = min(first + 1, int(channel_count) - 1)
        pairs.append(f"{first + 1} / {second + 1}")
    return pairs or ["1 / 1"]


def parse_channel_pair(value):
    return [int(part.strip()) - 1 for part in value.split("/")]


def endpoint_label(api_name, device_name):
    return f"[{HOSTAPI_LABELS.get(api_name, api_name)}] {device_name}"


def label_sort_key(label):
    if label.startswith("[JACK]"):
        priority = 0
    elif label.startswith("[ALSA]"):
        priority = 1
    else:
        priority = 2
    return priority, label.lower()


def build_endpoints(devices, hostapis, show_all=False):
    """Return (inputs, outputs) as label -> Endpoint, sorted JACK first.

    Raw ALSA hw:* devices are normally held open by PipeWire, so only its
    shared PCMs are offered unless ``show_all`` is set.  JACK devices with
    more than two channels are split into one endpoint per channel pair.
    """
    api_names = {}
    for hostapi in hostapis:
        for device_index in hostapi["devices"]:
            api_names[device_index] = hostapi["name"]
    inputs = {}
    outputs = {}
    for index, device in enumerate(devices):
        api_name = api_names.get(index, "")
        if not show_all and not (
            api_name == JACK_API
            or (api_name == ALSA_API and device["name"] in ALSA_SHARED_PCMS)
        ):
            continue
        for max_key, endpoints in (
            ("max_input_channels", inputs),
            ("max_output_channels", outputs),
        ):
            channel_count = int(device[max_key])
            if channel_count <= 0:
                continue
            label = endpoint_label(api_name, device["name"])
            if api_name != JACK_API:
                endpoints[label] = Endpoint(label, index, api_name, device["name"])
                continue
            pairs = channel_pairs(channel_count)
            for pair in pairs:
                pair_label = label if len(pairs) == 1 else f"{label} — {pair}"
                endpoints[pair_label] = Endpoint(
                    pair_label,
                    index,
                    api_name,
                    device["name"],
                    parse_channel_pair(pair),
                )
    return (
        dict(sorted(inputs.items(), key=lambda item: label_sort_key(item[0]))),
        dict(sorted(outputs.items(), key=lambda item: label_sort_key(item[0]))),
    )


class DeviceCatalog:
    def __init__(self, sd):
        self.sd = sd
        self.inputs = {}
        self.outputs = {}

    def refresh(self, show_all=False):
        # Re-initialize PortAudio so devices added since startup appear.
        self.sd._terminate()
        self.sd._initialize()
        self.inputs, self.outputs = build_endpoints(
            self.sd.query_devices(), self.sd.query_hostapis(), show_all
        )

    def default_label(self, direction):
        endpoints = self.inputs if direction == "input" else self.outputs
        default_index = self.sd.default.device[0 if direction == "input" else 1]
        return next(
            (label for label, ep in endpoints.items() if ep.index == default_index),
            next(iter(endpoints), ""),
        )

    def resolve(self, saved_label, direction):
        """Map a saved label to a current one, falling back to the default.

        A label can disappear when channel counts change, so a device with
        the same PortAudio name is preferred over the system default.
        """
        endpoints = self.inputs if direction == "input" else self.outputs
        if saved_label in endpoints:
            return saved_label
        if saved_label:
            saved_name = saved_label.split("] ", 1)[-1].split(" — ", 1)[0]
            match = next(
                (label for label, ep in endpoints.items() if ep.name == saved_name),
                None,
            )
            if match:
                return match
        return self.default_label(direction)

    def resolve_monitor(self, saved_label):
        if saved_label and saved_label in self.outputs:
            return saved_label
        return None

    def to_dict(self):
        def describe(endpoints):
            return [
                {
                    "label": ep.label,
                    "api": HOSTAPI_LABELS.get(ep.api, ep.api),
                    "name": ep.name,
                    "channels": [selector + 1 for selector in ep.selectors],
                }
                for ep in endpoints.values()
            ]

        return {"inputs": describe(self.inputs), "outputs": describe(self.outputs)}
