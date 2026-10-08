# Models

Create one folder for each RVC model under `models/`. The folder name is shown
in the application as the model name.

```text
models/
  MyModel/
    MyModel.pth
    added_MyModel.index
```

- Put the RVC model file (`.pth`) in the model folder.
- Put its retrieval index (`.index`) in the same folder. An `added_*.index`
  file is recommended.
- The folder name, for example `MyModel`, is displayed as the model name in the
  model list.

After copying a model folder, use **Reload** in the application to scan it.
