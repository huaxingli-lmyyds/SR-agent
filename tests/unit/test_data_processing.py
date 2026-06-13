from agent.data_processing.service import (
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    profile_dataset,
    publish_dataset_version,
)


def test_generic_dataset_profile_plan_and_publish(tmp_path, dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, target_goal="validate")
    results = execute_plan(plan)
    output = tmp_path / "versions" / "dataset.json"
    version = publish_dataset_version(dataset, results, output)

    assert profile.sample_count == 1
    assert results[-1].status == "success"
    assert version.dataset_id == dataset.dataset_id
    assert output.exists()
