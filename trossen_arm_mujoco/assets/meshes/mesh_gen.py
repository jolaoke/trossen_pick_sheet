import argparse
import shutil
from pathlib import Path

import trimesh


parser = argparse.ArgumentParser(description="Generate convex collision hulls from an STL mesh.")
parser.add_argument("filename", help="STL filename in the meshes directory")
parser.add_argument("--max-hulls", type=int, default=32, help="Maximum number of convex hulls to generate")
parser.add_argument("--resolution", type=int, default=100000, help="Resolution for V-HACD decomposition")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace the existing output directory and its contents",
)
args = parser.parse_args()

mesh_dir = Path(__file__).resolve().parent
input_path = Path(args.filename)
if not input_path.is_absolute():
    input_path = mesh_dir / input_path

output_dir = mesh_dir / f"collision_mesh_{input_path.stem}"
if output_dir.exists():
    if not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Re-run with --overwrite to replace it."
        )
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    shutil.rmtree(output_dir)

output_dir.mkdir()

# Load your high-resolution visual mesh
mesh = trimesh.load(input_path)

# Run V-HACD to generate a list of convex trimesh objects
# (Ensure your mesh is a single watertight object for best results)
convex_pieces = mesh.convex_decomposition( # pyright: ignore[reportAttributeAccessIssue]
    maxConvexHulls=args.max_hulls,
    resolution=args.resolution
)

# Export each convex hull as a separate STL for MuJoCo
# If it only returns one mesh (not a list), it will lack a length attribute, 
# so we force it into a list for consistent exporting.
if not isinstance(convex_pieces, list):
    convex_pieces = [convex_pieces]

for i, piece in enumerate(convex_pieces):
    piece.export(output_dir / f"collision_hull_{i}.stl")