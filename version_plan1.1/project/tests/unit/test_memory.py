from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService


def test_memory_isolated_by_task_model_and_dataset(tmp_path) -> None:
    service = MemoryService(tmp_path / "memory")
    service.remember_episode(
        EpisodeMemory(
            agent_type="hpo_agent",
            objective="improve ecapa",
            action={"best_config": {"lr": 0.001}},
            outcome={"best_metrics": {"eer": 0.03}},
            scope=MemoryScope(
                agent_type="hpo_agent",
                task_type="speaker_verification",
                model_family="ecapa_tdnn",
                dataset_key="dataset_a",
            ),
        ),
        force=True,
    )

    matched = service.get_model(
        "ecapa_tdnn",
        dataset_key="dataset_a",
        task_type="speaker_verification",
    )
    wrong_dataset = service.get_model(
        "ecapa_tdnn",
        dataset_key="dataset_b",
        task_type="speaker_verification",
    )

    assert matched["last_best_config"]["lr"] == 0.001
    assert wrong_dataset == {}
    assert len(service.search(MemoryQuery(model_family="ecapa_tdnn"))) == 1
