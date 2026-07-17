from pathlib import Path

PROJECT_DIRS = [
    "app",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "deployment",
    "docs",
    "models",
    "notebooks",
    "reports",
    "reports/figures",
    "reports/tables",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/pipelines",
    "src/api",
    "src/monitoring",
    "src/visualization",
    "src/utils",
    "tests",
    ".github/workflows",
]

PACKAGE_DIRS = [
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/pipelines",
    "src/api",
    "src/monitoring",
    "src/visualization",
    "src/utils",
]

FILES = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
    ".pre-commit-config.yaml",
    "ruff.toml",
]

GITKEEP_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "models",
    "reports/figures",
    "reports/tables",
]


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def main():
    for folder in PROJECT_DIRS:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for folder in PACKAGE_DIRS:
        touch(Path(folder) / "__init__.py")

    for file in FILES:
        touch(Path(file))

    for folder in GITKEEP_DIRS:
        touch(Path(folder) / ".gitkeep")

    print("✅ Project scaffold created successfully.")


if __name__ == "__main__":
    main()
