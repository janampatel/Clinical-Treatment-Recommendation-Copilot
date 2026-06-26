# Build (server-side) + deploy the Clinical Copilot to Cloud Run.
# No local Docker required — Cloud Build builds the image.
#
#   .\deploy\deploy.ps1 -ProjectId clinical-copilot-23
#
param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$Repo = "clinical-copilot",
  [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$Image = "$Region-docker.pkg.dev/$ProjectId/$Repo/copilot:$Tag"

Write-Host "==> Project: $ProjectId" -ForegroundColor Cyan
gcloud config set project $ProjectId | Out-Null
gcloud config set run/region $Region | Out-Null

# 1. Build the image in Cloud Build (uploads the build context, no local Docker).
Write-Host "==> Building image in Cloud Build: $Image" -ForegroundColor Cyan
gcloud builds submit --tag $Image .
if ($LASTEXITCODE -ne 0) {
  Write-Host "Cloud Build FAILED — not deploying. See the build log URL above." -ForegroundColor Red
  exit 1
}

# 2. Deploy the Streamlit UI (session affinity for websockets).
Write-Host "==> Deploying UI (copilot-ui)" -ForegroundColor Cyan
gcloud run deploy copilot-ui `
  --image $Image `
  --region $Region `
  --allow-unauthenticated `
  --memory 4Gi --cpu 2 --timeout 600 `
  --max-instances 2 --session-affinity `
  --set-env-vars "SERVICE=ui,COPILOT_SOURCE=processed,COPILOT_EMBEDDER=pubmedbert,LLM_MODE=template"

# 3. Deploy the FastAPI backend.
Write-Host "==> Deploying API (copilot-api)" -ForegroundColor Cyan
gcloud run deploy copilot-api `
  --image $Image `
  --region $Region `
  --allow-unauthenticated `
  --memory 4Gi --cpu 2 --timeout 600 `
  --max-instances 2 `
  --set-env-vars "SERVICE=api,COPILOT_SOURCE=processed,COPILOT_EMBEDDER=pubmedbert,LLM_MODE=template"

Write-Host "`n==> Done. URLs:" -ForegroundColor Green
gcloud run services describe copilot-ui  --region $Region --format "value(status.url)"
gcloud run services describe copilot-api --region $Region --format "value(status.url)"

Write-Host "`nNote: first request to each service is a cold start (~20-30s while it" -ForegroundColor Yellow
Write-Host "loads PubMedBERT + builds the index). Subsequent requests are fast." -ForegroundColor Yellow
