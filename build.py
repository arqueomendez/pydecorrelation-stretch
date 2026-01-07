import os
import subprocess
import sys

# tomllib is available in Python 3.11+
try:
    import tomllib
except ImportError:
    # Fallback for older python if needed (though project req is >=3.11 likely)
    # But since we run with uv python, we can enforce 3.11
    import tomllib  # type: ignore

from pathlib import Path


def get_project_version(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    # Type safe access
    project = data.get("project", {})
    version = project.get("version", "0.0.0")
    return str(version)


def build() -> None:
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"
    dist_path = project_root / "dist"
    pyproject_path = project_root / "pyproject.toml"

    version = get_project_version(pyproject_path)
    output_filename = f"DStretch-GUI-v{version}.exe"

    print(f"Detected project version: {version}")
    print(f"Target executable: {output_filename}")

    # Create a launcher script
    launcher_path = project_root / "launcher.py"
    with open(launcher_path, "w") as f:
        f.write("import sys\n")
        f.write("from pathlib import Path\n")
        f.write("from dstretch.gui import main\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    main()\n")

    print(f"Created launcher script at {launcher_path}")

    env = os.environ.copy()
    # Add src to PYTHONPATH so Nuitka can find 'dstretch' package
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) + os.pathsep + existing_pythonpath

    # Nuitka command
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "nuitka",
        "--standalone",
        "--onefile",
        "--enable-plugin=tk-inter",
        "--enable-plugin=numpy",
        "--include-package=dstretch",
        "--include-package-data=dstretch",
        "--assume-yes-for-downloads",  # Important for CI
        "--output-dir=dist",
        f"--output-filename={output_filename}",
        "--remove-output",
        str(launcher_path),
    ]

    print(f"Running build command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)

        if result.returncode == 0:
            print("\nBuild successful!")
            exe_path = dist_path / output_filename
            if exe_path.exists():
                print(f"Executable created at: {exe_path}")
            else:
                print("Warning: Executable not found despite success code.")
        else:
            print("\nBuild failed.")
            sys.exit(result.returncode)

    finally:
        # Cleanup launcher
        if launcher_path.exists():
            launcher_path.unlink()
            print(f"Cleaned up {launcher_path}")


if __name__ == "__main__":
    build()
