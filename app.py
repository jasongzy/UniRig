import os
import subprocess
import tempfile

import gradio as gr


def get_examples() -> list[list[str]]:
    cwd = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(cwd, "examples")
    if not os.path.exists(example_dir):
        return []
    extensions = (".obj", ".fbx", ".glb")
    return [
        [os.path.join(example_dir, f)] for f in os.listdir(example_dir) if any(f.endswith(ext) for ext in extensions)
    ]


def run_cmd(cmd: list[str], cwd: str) -> tuple[bool, str]:
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs = []
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
            logs.append(line)
    process.wait()

    full_log = "".join(logs)
    return process.returncode == 0, full_log


def process_pipeline(input_file: str, output_format: str, progress=gr.Progress()) -> str:
    if not input_file:
        raise gr.Error("Please upload a file first.")

    input_name = os.path.basename(input_file)
    name_no_ext = os.path.splitext(input_name)[0]
    root_dir = os.path.dirname(os.path.abspath(__file__))

    temp_dir = tempfile.mkdtemp(prefix="unirig_pipeline_")
    output_prefix = os.path.join(temp_dir, name_no_ext)

    skel_output = f"{output_prefix}_skel.fbx"
    skin_output = f"{output_prefix}_skin.fbx"
    rigged_output = f"{output_prefix}_rigged.{output_format.lower()}"

    progress(0.3, desc="Stage 1/3: Generating Skeleton...")
    success, log = run_cmd(
        ["bash", "launch/inference/generate_skeleton.sh", "--input", input_file, "--output", skel_output],
        root_dir,
    )
    if success:
        gr.Info("Skeleton generation completed.")
    else:
        raise gr.Error(f"Skeleton generation failed:\n{log}")

    progress(0.6, desc="Stage 2/3: Generating Skinning...")
    success, log = run_cmd(
        ["bash", "launch/inference/generate_skin.sh", "--input", skel_output, "--output", skin_output],
        root_dir,
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
            skin_output,
            "--target",
            input_file,
            "--output",
            rigged_output,
        ],
        root_dir,
    )
    if success:
        gr.Info("Merging completed.")
    else:
        raise gr.Error(f"Merging failed:\n{log}")

    progress(1.0, desc="Done!")
    return rigged_output


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    with gr.Blocks(title="UniRig") as app:
        gr.Markdown("# UniRig: One Model to Rig Them All")
        gr.Markdown("[Code](https://github.com/VAST-AI-Research/UniRig)")

        with gr.Row():
            with gr.Column():
                input_file = gr.File(label="Input Mesh", file_types=[".obj", ".fbx", ".glb"])
                output_format = gr.Radio(label="Output Format", choices=["glb", "fbx"], value="glb")
                run_btn = gr.Button("Run Rigging", variant="primary")

                gr.Examples(examples=get_examples(), inputs=input_file, label="Examples")

            with gr.Column():
                output_file = gr.File(label="Rigged Mesh")

        run_btn.click(
            fn=lambda: None,
            outputs=output_file,
        ).success(
            fn=process_pipeline,
            inputs=[input_file, output_format],
            outputs=output_file,
        )

    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
