from pathlib import Path

EXCLUDE = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
}


def tree(directory: Path, prefix: str = "") -> None:
    entries = sorted(
        [e for e in directory.iterdir() if e.name not in EXCLUDE],
        key=lambda e: (e.is_file(), e.name.lower()),
    )

    pointers = ["├── "] * (len(entries) - 1) + ["└── "]

    for pointer, entry in zip(pointers, entries):
        print(prefix + pointer + entry.name)

        if entry.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            tree(entry, prefix + extension)


if __name__ == "__main__":
    root = Path(".")
    print(root.resolve().name)
    tree(root)
