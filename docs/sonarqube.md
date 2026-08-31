# SonarQube analysis

The `SonarQube` CI job analyzes pushes to `main` and pull requests opened from
this repository. External-fork and Dependabot pull requests are skipped because
GitHub does not expose repository secrets to those workflows.

Configure these repository settings before enabling the job as a required
check:

- Variable `SONAR_HOST_URL`: `https://sonarcloud.io` for the configured
  SonarQube Cloud organization.
- Secret `SONAR_TOKEN`: a SonarQube Cloud token authorized to analyze the
  `meymchen_banshu` project.

The runner must be able to reach `SONAR_HOST_URL`. The job generates an LCOV
report for the complete Rust workspace with `cargo-llvm-cov`, then imports it
through `sonar.rust.lcov.reportPaths` from `sonar-project.properties`.
