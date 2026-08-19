# Publishing to PyPI

## One time setup

Your one manual step — register the trusted publisher on PyPI (requires your login):

1. Go to https://pypi.org/manage/account/publishing/ (you'll need 2FA enabled on the account).
2. Under "Add a new pending publisher" → GitHub tab, enter exactly:
  - PyPI project name: deep-research-harness
  - Owner: jmlon
  - Repository name: deep-research-harness
  - Workflow name: publish.yml
  - Environment name: pypi

Then to release 0.1.0:

cd ~/GIT/deep-research-harness
git tag v0.1.0
git push origin v0.1.0

## Deploying new releases

uv version --bump patch
uv version --bump minor
git commit,
git tag v<new-version>
git push && git push origin v<new-version>
