app-title = RVC リアルタイム

## Drag and drop
drop-title = ドロップしてモデルを追加、または声を測定
drop-detail = .pth ごとに1つのモデルになります。.index も一緒にドロップしてください。音声クリップやそのフォルダーをドロップすると、選択中のモデルの声のピッチを測定します。
model-imported = モデルを追加しました: { $names }

## Typed values
value-click-to-type = クリックして値を入力
value-invalid = 数値を入力してください

## Navigation
page-model = モデル
page-library = ライブラリ
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
assets-missing-banner = 推論用アセットが見つかりません: { $paths }。CUDA 12.8 リリースパッケージの assets/ フォルダーをプロジェクトフォルダーにコピーしてから再起動してください。
section-models = モデル
model = モデル
models-empty = モデルがありません。.pth ファイル（.index があれば一緒に）をこのウィンドウにドラッグして追加してください。
model-files = モデル: { $model } / Index: { $index }
none = なし
reload = 再読み込み
section-voice = 声
pitch = ピッチ（半音）
pitch-detail = 声をどれだけ高く/低くするか。+12 で 1 オクターブ上がります。下の「ピッチ合わせ」で目安を提案できます。
pitch-match = ピッチ合わせ
pitch-match-no-source = モデルの声: 未測定。その声のクリップ（できれば学習に使った音声）をここにドロップしてください。
pitch-match-analyzing = モデルの声: 測定中…
pitch-match-source = モデルの声: 中央値 { $hz } Hz
pitch-match-voice = あなたの声: 中央値 { $hz } Hz（発話 { $minutes } 分）
pitch-match-listening = あなたの声: 変換中に話すと測定されます（{ $seconds } / { $needed } 秒）
pitch-match-apply = 適用
pitch-match-measure = クリップを測定…
pitch-match-reset-voice = 測定したあなたの声をリセットしてやり直す
pick-pitch-clips = モデルの声のクリップを選択
output-level = 出力レベル
output-level-idle = 変換中に、相手にどのくらいの音量で聞こえるかを表示します。ボイスチャットは -21 LUFS、配信は配信音声の一般的なレベルである -16 LUFS を目標にします。
output-level-listening = 話すとレベルを測定します。
output-level-measuring = 現在 { $lufs } LUFS、測定中（発話 { $seconds } / 5 秒）
output-level-reading = { $lufs } LUFS（目標 { $target }）: { $verdict }
loudness-target-voice-chat = ボイスチャット（-21）
loudness-target-streaming = 配信（-16）
output-level-good = 適切なレベルです。
output-level-quiet = 小さすぎます。出力ゲインを上げてください。
output-level-quiet-peaky = 小さめですが、ピークがすでにフルスケールに達しているため、ゲインを上げると潰れてしまいます。
output-level-loud = 大きすぎます。出力ゲインを下げてください。
output-level-squashed = 大きすぎます。ピークが潰れています。出力ゲインを下げてください。
output-level-apply = 適用
level-good = レベル: 適切
level-quiet = レベル: 小さすぎ
level-loud = レベル: 大きすぎ
level-measuring = レベル: 測定中…
library-search = モデルを検索
library-sort-name = 名前順
library-sort-newest = 新しい順
library-count = { $total } 件中 { $shown } 件
library-no-match = 検索に一致するモデルはありません。
library-voice = { $voice }（{ $count }）
library-checkpoint = { $detail } · { $epochs } エポック
library-epochs = { $epochs } エポック
library-index-yes = インデックスあり
library-index-no = インデックスなし
library-use = 使用
library-selected = 選択中
library-open-folder = フォルダーを開く
library-delete = ゴミ箱へ移動
library-delete-in-use = このモデルを削除するには変換を停止してください
delete-model-title = モデルをゴミ箱へ移動しますか？
delete-model-body = { $name } と保存された設定（ピッチ、ゲイン）がゴミ箱へ移動されます。ゴミ箱から復元できます。
delete-model-confirm = ゴミ箱へ移動
cancel = キャンセル
status-model-deleted = { $name } をゴミ箱へ移動しました
error-model-in-use = このモデルを削除する前に変換を停止してください。
error-delete-failed = モデルをゴミ箱へ移動できませんでした: { $detail }
library-rename = 名前を変更
library-rename-in-use = このモデルの名前を変更するには変換を停止してください
rename-model-title = モデル名の変更
rename-model-body = モデルのフォルダー名を変更します。保存された設定はそのまま引き継がれます。
rename-model-confirm = 名前を変更
status-model-renamed = { $name } を { $new_name } に変更しました
error-model-name-taken = その名前のモデルはすでにあります。
error-bad-model-name = その名前は使えません。空の名前、ドットで始まる名前、/ \ : * ? " < > | を含む名前は使えません。
error-rename-failed = モデルの名前を変更できませんでした: { $detail }
formant = フォルマント
formant-detail = ピッチを変えずに声の響き（話者の体格の印象）を変えます。上げると小柄で明るく、下げると大柄で太く聞こえます。0 はモデル本来の響きです。
index-rate = インデックス使用率
index-rate-detail = .index ファイルを使って、声をモデルの元の録音に近づけます。上げると話者に近づきますが、その人のなまりや癖も入ります。0 でオフです。
protect = 子音の保護
protect-detail = インデックス使用率が 0 より大きいときだけ有効です。息や子音（s、t、k）を元の声に近く保ち、にじみを防ぎます。低いほど強く保護します。
section-levels = レベル
input-gain = 入力ゲイン
input-gain-detail = 変換前のマイク音量を上げ下げします。変換後の声は入力の大きさに追従するため、出力の大きさも変わります。
output-gain = 出力ゲイン
output-gain-detail = 出力デバイスに送る変換後の声の音量（相手に聞こえる音量）です。下の「出力レベル」が目安を提案します。
monitor-gain = モニターゲイン
monitor-gain-detail = モニターデバイスで自分が聞く音量です。相手に聞こえる音量は変わりません。
noise-gate = ノイズゲート
noise-gate-detail = マイクがこの値より小さいときは無音にし、言葉の合間の環境音を変換しないようにします。-60 dB でオフです。
off = オフ
reset-model-settings = モデル設定をリセット

## Audio page
section-input = 入力
input-source = 入力元
input-source-detail = マイクをリアルタイムで変換するか、音声ファイルをモデルに通して再生します。
source-microphone = マイク／音声デバイス
source-file = 音声ファイル
audio-file = ファイル
no-file = ファイル未選択
browse = 参照…
pick-audio-file = 音声ファイルを選択
file-volume = ファイル音量
file-volume-detail = 変換したファイルの再生音量です。
input-device = 入力デバイス
input-device-detail = 使用するマイクです。
section-output = 出力
output-device = 出力デバイス
output-device-detail = 変換後の声の送り先です。通常は Discord や OBS が聞く仮想マイクです。
monitor-device = モニターデバイス
monitor-device-detail = 自分の声を聞くための任意の 2 つ目の出力（ヘッドホンなど）です。
monitor-disabled = 使用しない
reload-devices = デバイスリストのリロード
sample-rate-value = 動作サンプルレート：{ $rate } Hz
sample-rate-idle = 動作サンプルレート：—（停止中）
jack-hint = [JACK] デバイスが最も低遅延です。周期は PIPEWIRE_QUANTUM（例: 128/48000）で設定します。

## Performance page
section-buffering = バッファー
chunk = チャンク長
chunk-detail = 一度に変換する音声の長さです。長くした分だけ遅延が増えます。短くすると遅延は減りますが、GPU が追いつかないと途切れます（下部バーの「推論」を確認してください）。変更すると変換が再起動します。
extra = 追加推論バッファ
extra-detail = 各チャンクを変換するときに、モデルが参照する直前の音声の長さです。長いほど声質とピッチが安定し、GPU 負荷はわずかに増えますが、遅延は増えません。変更すると変換が再起動します。
section-inference = 推論
pitch-detector = ピッチ検出方式
pitch-detector-detail = ピッチの測り方です。RMVPE が最も正確、FCPE は軽量、PM は最速ですが最も粗いです。
pitch-smoothing = ピッチの平滑化
pitch-smoothing-detail = 10 ms ごとに個別に判定せず、ピッチを一本の連続した線として追跡します。伸ばした音が安定し、オクターブの跳びがなくなります。遅延は増えません。
pitch-smoothing-rmvpe-only = RMVPE ピッチ検出方式でのみ有効です。
f0-rmvpe = RMVPE（推奨）
f0-fcpe = FCPE
f0-pm = PM（最速）
volume-envelope = 音量追従ミックス
volume-envelope-detail = 変換後の声の大きさを、瞬間ごとにどれだけあなたの声に合わせるか。0 はあなたの音量に完全に合わせ、1 はモデル本来の音量を保ちます。
gpu = 使用GPU
gpu-detail = モデルを実行するグラフィックカードです。変更すると変換が再起動します。
gpu-automatic = 自動（推奨）
hold-context = 無音中はコンテキストを保持
hold-context-detail = 黙っている間も直前の発話をモデルの文脈として保持し、間の後の最初の言葉が不明瞭になるのを防ぎます。
input-denoise-rnnoise = 変換前にマイクのキーボード音、ファン、車の音などの雑音を除去します。RNNoise を使用し、遅延が 20 ms 増えます。
input-denoise-spectral = 変換前にマイクの雑音を除去します。スペクトルゲートを使用し、遅延が 40 ms 増えます。RNNoise（sudo pacman -S rnnoise）を導入すると、より少ない遅延でよりよく除去できます。
hold-detector = 無音の判定
hold-detector-level = 音量
hold-detector-voice = 音声検出（実験的）
hold-detector-level-detail = -50 dBFS（またはノイズゲート）未満を無音とみなします。静かな部屋向けです。
hold-detector-voice-detail = RNNoise が発話かどうかを判定するため、ファンやハムノイズを発話とみなしません。騒がしい部屋向けです。
section-noise = ノイズ低減
input-denoise = 入力ノイズの低減
output-denoise = 出力ノイズの低減
output-denoise-detail = 変換後の声のヒスノイズをスペクトルゲートで除去します。声が少しこもることがあるため、ヒスが聞こえる場合だけオンにしてください。
reset-performance = パフォーマンス設定をリセット

## Recording page
section-recording = 録音
recording-mode = モード
recording-mode-detail = 録音にはマイクの元の音声と変換後の声が含まれます。保存方法を選んでください。
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
latency-detail = マイクから出力までの遅延（チャンク、クロスフェード、デバイスバッファ）に、現在キューにある音声を加えたものです。急に増えた場合はエンジンが一時的に遅れています。次の無音の間に追いつきます。実際の遅延はこれより少し大きくなります。
inference = 推論: { $value } ms
inference-detail = 各チャンクの変換にかかる時間です。チャンク長より十分短くないと音が途切れます。

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
status-analyzing-pitch = { $name } の声のピッチを測定しています…
status-pitch-analyzed = { $name }: 中央値 { $hz } Hz（発話 { $minutes } 分）
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
error-assets-missing = 開始できません: 推論用アセットがありません（{ $paths }）。リリースパッケージの assets/ フォルダーをプロジェクトフォルダーにコピーしてください。
error-model-load-failed = モデルを読み込めませんでした: { $detail }
error-no-voice-found = クリップに有声の発話が見つかりませんでした。
error-pitch-analysis-busy = 別のクリップを測定中です。終わるまでお待ちください。
error-pitch-analysis-failed = クリップを測定できませんでした: { $detail }
