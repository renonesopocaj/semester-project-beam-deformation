import os
import pandas as pd
import wandb

class WandbRunExporter:

    def __init__(self, project_path, output_dir):
        """
        Args:
            project_path: str
                W&B path in the form "entity/project"
            output_dir: str
                Local directory where all runs are exported
            return: None
                Initializes the exporter
        """
        self.api = wandb.Api()
        self.project_path = project_path
        self.output_dir = output_dir

    def export_all_runs(self):
        """
        Args:
            return: None
                Exports run files, history, and logged artifacts for every run
        """
        runs = self.api.runs(self.project_path)
        for run in runs:
            self.export_single_run(run)

    def export_single_run(self, run):
        """
        Args:
            run: wandb.apis.public.Run
                A W&B run object
            return: None
                Exports one run locally
        """
        run_dir = os.path.join(self.output_dir, run.id)
        os.makedirs(run_dir, exist_ok=True)

        self.export_run_files(run, run_dir)
        self.export_history(run, run_dir)
        self.export_summary(run, run_dir)
        self.export_logged_artifacts(run, run_dir)

    def export_run_files(self, run, run_dir):
        """
        Args:
            run: wandb.apis.public.Run
                A W&B run object
            run_dir: str
                Output directory for the run
            return: None
                Downloads files attached directly to the run
        """
        files_dir = os.path.join(run_dir, "files")
        os.makedirs(files_dir, exist_ok=True)

        for file_obj in run.files():
            local_path = os.path.join(files_dir, file_obj.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path):
                continue
            file_obj.download(root=files_dir, replace=False)

    def export_history(self, run, run_dir):
        """
        Args:
            run: wandb.apis.public.Run
                A W&B run object
            run_dir: str
                Output directory for the run
            return: None
                Saves full metric history as CSV
        """
        history_rows = list(run.scan_history())
        if len(history_rows) == 0:
            return

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    def export_summary(self, run, run_dir):
        """
        Args:
            run: wandb.apis.public.Run
                A W&B run object
            run_dir: str
                Output directory for the run
            return: None
                Saves summary metrics as CSV
        """
        summary_dict = dict(run.summary)
        if len(summary_dict) == 0:
            return

        summary_df = pd.DataFrame([summary_dict])
        summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    def export_logged_artifacts(self, run, run_dir):
        """
        Args:
            run: wandb.apis.public.Run
                A W&B run object
            run_dir: str
                Output directory for the run
            return: None
                Downloads artifacts produced by the run
        """
        artifacts_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        for artifact in run.logged_artifacts():
            artifact_name = artifact.name.replace("/", "_").replace(":", "_")
            target_dir = os.path.join(artifacts_dir, artifact_name)
            if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
                continue
            artifact.download(root=target_dir)


if __name__ == "__main__":
    exporter = WandbRunExporter("jacoposenoner/beam_deformation", "../wandb_export")
    exporter.export_all_runs()