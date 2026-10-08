app-title = RVC リアルタイム

## Drag and drop
drop-title = ドロップしてモデルを追加
drop-detail = .pth ごとに1つのモデルになります。.index も一緒にドロップしてください。
model-imported = モデルを追加しました: { $names }

## Navigation
page-model = モデル
page-audio = オーディオ
page-performance = パフォーマンス
page-recording = 録音
page-log = ログ

## Engine connection
engine-starting = エンジンを起動しています…
engine-starting-detail = PyTorch、モデル、オーディオデバイスを読み込んでいます。
engine-unavailable = エンジンを利用できません
engine-retrying = 自動的に再試行します。

## Model page
section-models = モデル
model = モデル
models-empty = モデルがありません。.pth ファイル（.index があれば一緒に）をこのウィンドウにドラッグして追加してください。
model-files = モデル: { $model } / Index: { $index }
none = なし
reload = 再読み込み
section-voice = 声
pitch = ピッチ（半音）
formant = フォルマント
index-rate = インデックス使用率
section-levels = レベル
input-gain = 入力ゲイン
output-gain = 出力ゲイン
monitor-gain = モニターゲイン
noise-gate = ノイズゲート
off = オフ
reset-model-settings = モデル設定をリセット

## Audio page
section-input = 入力
input-source = 入力元
source-microphone = マイク／音声デバイス
source-file = 音声ファイル
audio-file = ファイル
no-file = ファイル未選択
browse = 参照…
pick-audio-file = 音声ファイルを選択
file-volume = ファイル音量
input-device = 入力デバイス
section-output = 出力
output-device = 出力デバイス
monitor-device = モニターデバイス
monitor-disabled = 使用しない
reload-devices = デバイスリストのリロード
sample-rate-value = 動作サンプルレート：{ $rate } Hz
sample-rate-idle = 動作サンプルレート：—（停止中）
jack-hint = [JACK] デバイスが最も低遅延です。周期は PIPEWIRE_QUANTUM（例: 128/48000）で設定します。

## Performance page
section-buffering = バッファー
chunk = チャンク長
crossfade = クロスフェード
extra = 追加推論バッファ
section-inference = 推論
pitch-detector = ピッチ検出方式
f0-rmvpe = RMVPE（推奨）
f0-fcpe = FCPE
f0-pm = PM（最速）
volume-envelope = 音量追従ミックス
gpu = 使用GPU
gpu-automatic = 自動（推奨）
section-noise = ノイズ低減
input-denoise = 入力ノイズの低減
output-denoise = 出力ノイズの低減
reset-performance = パフォーマンス設定をリセット

## Recording page
section-recording = 録音
recording-mode = モード
recording-separate = 別ファイル（入力＋変換後）
recording-mix = ミックス（入力＋変換後）
recording-stereo = L/R分離（左: 入力／右: 変換後）
save-folder = 保存先
pick-recording-folder = 録音フォルダーを選択
change = 変更…
open-folder = フォルダを開く
record = 録音
stop-recording = 録音を停止
record-hint = 変換を開始してから録音してください。

## Log page
save-log = ログを保存
clear-log = ログを消去
log-saved = ログを保存しました: { $path }

## Control bar
start = スタート
starting = 起動中…
converting = 変換中
passthrough = 音声パススルー
passthrough-active = パススルー中
meter-in = 入力
meter-out = 出力
meter-mon = モニ
latency = 遅延: { $value } ms
inference = 推論: { $value } ms

## Engine status
status-preparing = 起動準備中…
status-loading-model = モデルを読み込んでいます…
status-preparing-inference = 推論を準備しています…
status-starting-audio = 音声デバイスを開始しています…
status-conversion-started = 変換を開始しました
status-passthrough-started = 音声パススルーを開始しました
status-conversion-stopped = 変換を停止しました
status-passthrough-stopped = 音声パススルーを停止しました
status-settings-changed = 設定変更のため変換を停止しました
status-stream-stopped = 音声デバイスが予期せず停止しました
status-file-selected = 音声ファイルを選択しました: { $name }
status-playing = 再生中
status-paused = 一時停止中
status-playback-stopped = 再生を停止しました
status-recording-started = 録音を開始しました
status-importing-model = モデルを追加しています…
status-recording-saved = 録音を保存しました:
    { $paths }

## Engine errors
error-no-model = modelsフォルダー内のモデルを選択してください。
error-model-file-missing = モデルファイルが見つかりません。
error-index-missing = 選択したモデルに.indexファイルがありません。インデックス使用率を0にするか、モデルフォルダーへ.indexを追加してください。
error-no-audio-file = 音声ファイルを選択してください。
error-ffmpeg-missing = FFmpegが見つかりません。ffmpeg をインストールしてください。
error-record-requires-running = 変換を開始してから録音してください。
error-no-common-samplerate = 入力・出力・モニターデバイスに共通のサンプルレートがありません。同じレート（通常は48 kHz）に設定してください。
error-audio-start-failed = 音声デバイスを開始できませんでした: { $detail }
error-monitor-failed = モニターデバイスを開始できませんでした。通常の出力のみで続行します。（{ $detail }）
error-not-ready = エンジンを起動しています。
error-import-no-model-file = RVCモデルの .pth ファイルをドロップしてください。.index も一緒にドロップできます。
error-import-failed = モデルを追加できませんでした: { $detail }
