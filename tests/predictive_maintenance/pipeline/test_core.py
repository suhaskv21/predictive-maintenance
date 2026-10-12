from predictive_maintenance.pipeline.core import Pipeline


class AddOne:
    def run(self, data: int) -> int:
        return data + 1


class Double:
    def run(self, data: int) -> int:
        return data * 2


def test_pipeline_runs_stages_in_the_configured_order() -> None:
    assert Pipeline([AddOne(), Double()]).run(3) == 8
    assert Pipeline([Double(), AddOne()]).run(3) == 7


def test_empty_pipeline_returns_its_input_unchanged() -> None:
    assert Pipeline[int]([]).run(5) == 5
