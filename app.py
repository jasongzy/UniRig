import os
import subprocess
import tempfile

import gradio as gr

CWD = os.path.dirname(os.path.abspath(__file__))


def get_examples() -> list[list[str]]:
    example_dir = os.path.join(CWD, "examples")
    if not os.path.exists(example_dir):
        return []
    extensions = (".obj", ".fbx", ".glb")
    return [
        [os.path.join(example_dir, f)] for f in os.listdir(example_dir) if any(f.endswith(ext) for ext in extensions)
    ]


def run_cmd(cmd: list[str], cwd: str = CWD) -> tuple[bool, str]:
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs = []
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
            logs.append(line)
    process.wait()

    return process.returncode == 0, "".join(logs)


def process_pipeline(input_path: str, output_format: str, progress=gr.Progress()) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    temp_dir = tempfile.mkdtemp(prefix="unirig_pipeline_")
    output_prefix = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_path))[0])

    skel_output_path = f"{output_prefix}_skel.fbx"
    skin_output_path = f"{output_prefix}_skin.fbx"
    rigged_output_path = f"{output_prefix}_rigged.{output_format.lower()}"

    progress(0.3, desc="Stage 1/3: Generating Skeleton...")
    success, log = run_cmd(
        ["bash", "launch/inference/generate_skeleton.sh", "--input", input_path, "--output", skel_output_path]
    )
    if success:
        gr.Info("Skeleton generation completed.")
    else:
        raise gr.Error(f"Skeleton generation failed:\n{log}")

    progress(0.6, desc="Stage 2/3: Generating Skinning...")
    success, log = run_cmd(
        ["bash", "launch/inference/generate_skin.sh", "--input", skel_output_path, "--output", skin_output_path]
    )
    if success:
        gr.Info("Skinning generation completed.")
    else:
        raise gr.Error(f"Skinning generation failed:\n{log}")

    progress(0.9, desc="Stage 3/3: Merging Results...")
    success, log = run_cmd(
        [
            "bash",
            "launch/inference/merge.sh",
            "--source",
            skin_output_path,
            "--target",
            input_path,
            "--output",
            rigged_output_path,
        ]
    )
    if success:
        gr.Info("Merging completed.")
    else:
        raise gr.Error(f"Merging failed:\n{log}")

    progress(1.0, desc="Done!")
    gr.Success("Pipeline finished successfully!")
    return rigged_output_path


if __name__ == "__main__":
    with gr.Blocks(title="UniRig") as app:
        gr.Markdown("# UniRig: One Model to Rig Them All")
        gr.Markdown("[Code](https://github.com/VAST-AI-Research/UniRig)")

        with gr.Row():
            with gr.Column():
                input_3d = gr.File(label="Input Mesh", file_types=[".obj", ".fbx", ".glb"])
                output_format = gr.Radio(label="Output Format", choices=["glb", "fbx"], value="glb")
                run_btn = gr.Button("Run", variant="primary")

                gr.Examples(examples=get_examples(), inputs=input_3d, label="Examples")

            with gr.Column():
                output_3d = gr.File(label="Rigged Mesh")

        run_btn.click(
            fn=lambda: None,
            outputs=output_3d,
        ).success(
            fn=process_pipeline,
            inputs=[input_3d, output_format],
            outputs=output_3d,
        )

    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
