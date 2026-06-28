# rPPG-Emotion pipeline runner (PowerShell). Run steps end to end.
# Usage:  .\run.ps1 <step>
#   setup     environment check
#   data      build HRV dataset from WESAD (downloads+extracts if needed)
#   train     XGBoost LOSO-CV  (+ optional MLP)
#   evaluate  feature importance, SHAP, arousal-vs-valence
#   verify    full accuracy gate
#   visuals   content figures
#   web       launch the browser webcam demo (Flask, http://localhost:5000)
#   demo      launch live Gradio app
#   all       setup -> data -> train -> evaluate -> verify -> visuals

param([string]$step = "all")
$py = ".\.venv\Scripts\python.exe"

function Run($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan; & $py -m $m; if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 } }

switch ($step) {
  "setup"    { Run "verification.setup_check" }
  "data"     { Run "data.load_wesad" }
  "train"    { Run "emotion.train" }
  "evaluate" { Run "emotion.evaluate" }
  "verify"   { Run "verification.full_accuracy_check" }
  "visuals"  { Run "demo.generate_visuals" }
  "web"      { & $py -m demo.web_app }
  "demo"     { & $py -m demo.gradio_app }
  "all" {
    Run "verification.setup_check"
    Run "data.load_wesad"
    Run "emotion.train"
    Run "emotion.evaluate"
    Run "verification.full_accuracy_check"
    Run "demo.generate_visuals"
    Write-Host "`nDone. Launch demo with: .\run.ps1 demo" -ForegroundColor Green
  }
  default { Write-Host "unknown step '$step'. Use: setup|data|train|evaluate|verify|visuals|web|demo|all" }
}
