class DistMetrics():
    def __init__(self, o):
        super().__init__(o)

        self.run = self.create_wandb_logs(o)
        self.run_table = self.create_run_table(o)