"""Streamlit research UI for DBP prediction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from dbp_prediction.ui.configurator import (
    DEFAULT_FEATURE_STEPS_TEXT,
    DEFAULT_OUTPUT_DIR,
    PARAM_LABELS,
    TUNING_PRESETS,
    FieldSpec,
    build_experiment_payload,
    build_model_entry,
    default_dataset_path,
    default_feature_columns,
    default_model_params,
    default_search_form_state,
    default_split_column,
    default_target_columns,
    default_training_params,
    get_model_field_specs,
    get_model_label,
    get_model_search_space,
    get_training_field_specs,
    list_transform_names,
    ordered_model_names,
    parse_feature_steps,
    parse_list_of_ints,
    parse_optional_number,
    persist_uploaded_dataset,
    preview_dataset,
    render_config_preview,
    estimate_training_load,
)
from dbp_prediction.ui.runtime import execute_payload

APP_TITLE = "DBP Research Workbench"


def _key(*parts: str) -> str:
    return "__".join(parts)


def _format_choice(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _field_default_value(field: FieldSpec) -> Any:
    if field.kind == "int_list":
        return ",".join(str(item) for item in field.default)
    if field.kind in {"optional_int", "optional_float"}:
        return "" if field.default is None else str(field.default)
    return field.default


def _ensure_base_state() -> None:
    st.session_state.setdefault("dataset_source_mode", "Use an existing file path")
    st.session_state.setdefault("dataset_path", default_dataset_path())
    st.session_state.setdefault("dataset_format", "csv")
    st.session_state.setdefault("feature_steps_text", DEFAULT_FEATURE_STEPS_TEXT)
    st.session_state.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))
    st.session_state.setdefault("save_models", True)
    st.session_state.setdefault("save_predictions", False)
    st.session_state.setdefault("prepare_only_inspect_data", True)
    st.session_state.setdefault("run_id", "")
    st.session_state.setdefault("tuning_enabled", False)
    st.session_state.setdefault("tuning_preset", "Standard")
    preset = TUNING_PRESETS["Standard"]
    st.session_state.setdefault("global_trials", preset["trials"])
    st.session_state.setdefault("global_folds", preset["folds"])
    st.session_state.setdefault("global_stability_penalty", preset["stability_penalty"])
    st.session_state.setdefault("tuning_scout", False)

    for field in get_training_field_specs():
        st.session_state.setdefault(_key("training", field.key), _field_default_value(field))

    for index, model_name in enumerate(ordered_model_names()):
        st.session_state.setdefault(_key("model", model_name, "enabled"), index < 2)
        st.session_state.setdefault(_key("model", model_name, "alias"), "")
        st.session_state.setdefault(_key("model", model_name, "use_global_tuning"), True)
        st.session_state.setdefault(_key("model", model_name, "use_default_search"), True)
        st.session_state.setdefault(_key("model", model_name, "trials"), TUNING_PRESETS["Standard"]["trials"])
        st.session_state.setdefault(_key("model", model_name, "folds"), TUNING_PRESETS["Standard"]["folds"])
        st.session_state.setdefault(
            _key("model", model_name, "stability_penalty"),
            TUNING_PRESETS["Standard"]["stability_penalty"],
        )

        for field in get_model_field_specs(model_name):
            st.session_state.setdefault(
                _key("model", model_name, "params", field.key),
                _field_default_value(field),
            )

        search_state = default_search_form_state(model_name)
        for group_name in ("model", "training"):
            for param_name, param_state in search_state[group_name].items():
                base = _key("search", model_name, group_name, param_name)
                st.session_state.setdefault(f"{base}__enabled", param_state["enabled"])
                st.session_state.setdefault(f"{base}__choices", param_state["choices"])
                if param_state["low"] is not None:
                    st.session_state.setdefault(f"{base}__low", param_state["low"])
                if param_state["high"] is not None:
                    st.session_state.setdefault(f"{base}__high", param_state["high"])
                if param_state["step"] is not None:
                    st.session_state.setdefault(f"{base}__step", param_state["step"])
                st.session_state.setdefault(f"{base}__log", param_state["log"])

        for study_key, study_value in search_state["study"].items():
            st.session_state.setdefault(_key("study", model_name, study_key), study_value)


def _apply_tuning_preset() -> None:
    preset_name = st.session_state["tuning_preset"]
    preset = TUNING_PRESETS.get(preset_name)
    if preset is None:
        return
    st.session_state["global_trials"] = preset["trials"]
    st.session_state["global_folds"] = preset["folds"]
    st.session_state["global_stability_penalty"] = preset["stability_penalty"]


def _render_field(field: FieldSpec, prefix: str) -> None:
    widget_key = _key(prefix, field.key)
    if field.kind == "int":
        st.number_input(
            field.label,
            min_value=int(field.min_value) if field.min_value is not None else None,
            max_value=int(field.max_value) if field.max_value is not None else None,
            step=int(field.step) if field.step is not None else 1,
            key=widget_key,
            help=field.help_text,
        )
        return

    if field.kind == "float":
        st.number_input(
            field.label,
            min_value=float(field.min_value) if field.min_value is not None else None,
            max_value=float(field.max_value) if field.max_value is not None else None,
            step=float(field.step) if field.step is not None else 0.01,
            format="%.6f",
            key=widget_key,
            help=field.help_text,
        )
        return

    if field.kind == "bool":
        st.checkbox(field.label, key=widget_key, help=field.help_text)
        return

    if field.kind == "select":
        st.selectbox(
            field.label,
            options=list(field.choices),
            format_func=_format_choice,
            key=widget_key,
            help=field.help_text,
        )
        return

    if field.kind in {"text", "optional_int", "optional_float", "int_list"}:
        st.text_input(field.label, key=widget_key, help=field.help_text)
        return

    raise ValueError(f"Unsupported field kind: {field.kind}")


def _collect_field_values(field_specs: tuple[FieldSpec, ...], prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in field_specs:
        widget_key = _key(prefix, field.key)
        raw_value = st.session_state.get(widget_key)
        if field.kind == "int_list":
            values[field.key] = parse_list_of_ints(str(raw_value))
        elif field.kind == "optional_int":
            values[field.key] = parse_optional_number(str(raw_value), as_type=int)
        elif field.kind == "optional_float":
            values[field.key] = parse_optional_number(str(raw_value), as_type=float)
        else:
            values[field.key] = raw_value
    return values


def _render_search_editor(model_name: str) -> None:
    search_space = get_model_search_space(model_name)
    for group_name in ("model", "training"):
        group = search_space.get(group_name, {})
        if not group:
            continue
        st.markdown(f"**{group_name.title()} Search Parameters**")
        for param_name, spec in group.items():
            label = PARAM_LABELS.get(param_name, param_name.replace("_", " ").title())
            base = _key("search", model_name, group_name, param_name)
            with st.container(border=True):
                cols = st.columns([1.6, 1.1, 1.3])
                cols[0].markdown(f"`{label}`")
                cols[1].checkbox("Include in search", key=f"{base}__enabled")
                cols[2].caption(f"Type: {str(spec.get('type', 'categorical')).title()}")

                if spec.get("when"):
                    conditions = ", ".join(
                        f"{name} = {_format_choice(value)}"
                        for name, value in dict(spec["when"]).items()
                    )
                    st.caption(f"Condition: only active when {conditions}.")

                if spec.get("type") == "categorical":
                    st.multiselect(
                        "Candidates",
                        options=list(spec.get("choices", [])),
                        format_func=_format_choice,
                        key=f"{base}__choices",
                    )
                    continue

                value_cols = st.columns(4)
                value_cols[0].number_input(
                    "Min",
                    key=f"{base}__low",
                    format="%.6f" if spec.get("type") == "float" else "%d",
                )
                value_cols[1].number_input(
                    "Max",
                    key=f"{base}__high",
                    format="%.6f" if spec.get("type") == "float" else "%d",
                )
                if spec.get("step") is not None:
                    value_cols[2].number_input(
                        "Step",
                        key=f"{base}__step",
                        format="%.6f" if spec.get("type") == "float" else "%d",
                    )
                if spec.get("type") == "float":
                    value_cols[3].checkbox("Log scale", key=f"{base}__log")

    if search_space.get("study"):
        with st.expander("Advanced Study Settings", expanded=False):
            for study_key in search_space["study"]:
                st.number_input(
                    PARAM_LABELS.get(study_key, study_key.replace("_", " ").title()),
                    min_value=0,
                    step=1,
                    key=_key("study", model_name, study_key),
                )


def _collect_search_form_state(model_name: str) -> dict[str, Any]:
    default_state = default_search_form_state(model_name)
    collected: dict[str, Any] = {"model": {}, "training": {}, "study": {}}

    for group_name in ("model", "training"):
        for param_name, param_state in default_state[group_name].items():
            base = _key("search", model_name, group_name, param_name)
            current = {
                "enabled": bool(st.session_state[f"{base}__enabled"]),
                "type": param_state["type"],
                "choices": list(st.session_state.get(f"{base}__choices", [])),
                "low": st.session_state.get(f"{base}__low"),
                "high": st.session_state.get(f"{base}__high"),
                "step": st.session_state.get(f"{base}__step"),
                "log": bool(st.session_state.get(f"{base}__log", False)),
                "when": param_state.get("when"),
                "default": param_state.get("default"),
            }

            if current["type"] == "int":
                if current["low"] is not None:
                    current["low"] = int(current["low"])
                if current["high"] is not None:
                    current["high"] = int(current["high"])
                if current["step"] is not None:
                    current["step"] = int(current["step"])
            elif current["type"] == "float":
                if current["low"] is not None:
                    current["low"] = float(current["low"])
                if current["high"] is not None:
                    current["high"] = float(current["high"])
                if current["step"] is not None:
                    current["step"] = float(current["step"])

            collected[group_name][param_name] = current

    for study_key in default_state["study"]:
        collected["study"][study_key] = int(st.session_state[_key("study", model_name, study_key)])

    return collected


def _sync_column_defaults(columns: list[str]) -> None:
    default_features = [column for column in st.session_state.get("feature_columns", []) if column in columns]
    default_targets = [column for column in st.session_state.get("target_columns", []) if column in columns]
    default_selected_targets = [
        column for column in st.session_state.get("selected_targets", []) if column in columns
    ]

    if not default_features:
        default_features = default_feature_columns(columns)
    if not default_targets:
        default_targets = default_target_columns(columns)
    if not default_selected_targets:
        default_selected_targets = list(default_targets)

    st.session_state["feature_columns"] = default_features
    st.session_state["target_columns"] = default_targets
    st.session_state["selected_targets"] = default_selected_targets

    split_column = st.session_state.get("split_column")
    if split_column not in columns:
        st.session_state["split_column"] = default_split_column(columns)


def _build_current_model_entries(tuning_enabled: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for model_name in ordered_model_names():
        params = _collect_field_values(get_model_field_specs(model_name), _key("model", model_name, "params"))
        entry = build_model_entry(
            model_name=model_name,
            enabled=bool(st.session_state[_key("model", model_name, "enabled")]),
            alias=str(st.session_state[_key("model", model_name, "alias")]),
            params=params,
            tuning_enabled=tuning_enabled,
            use_global_tuning=bool(st.session_state[_key("model", model_name, "use_global_tuning")]),
            model_tuning={
                "trials": int(st.session_state[_key("model", model_name, "trials")]),
                "folds": int(st.session_state[_key("model", model_name, "folds")]),
                "stability_penalty": float(
                    st.session_state[_key("model", model_name, "stability_penalty")]
                ),
            },
            use_default_search_space=bool(
                st.session_state[_key("model", model_name, "use_default_search")]
            ),
            search_form_state=_collect_search_form_state(model_name),
        )
        entries.append(entry)
    return entries


def _render_execution_result() -> None:
    result = st.session_state.get("last_execution")
    if result is None:
        return

    st.success(f"Run `{result.run_id}` completed in `{result.mode}` mode.")
    st.write(f"Output directory: `{result.output_dir}`")
    st.write(f"Summary: `{result.summary_path}`")
    st.write(f"Plan: `{result.plan_path}`")
    st.write(f"Resolved config: `{result.config_snapshot_path}`")
    if result.manifest_path is not None:
        st.write(f"Manifest: `{result.manifest_path}`")
    if result.comparison_path is not None:
        st.write(f"Comparison: `{result.comparison_path}`")
        try:
            comparison_payload = json.loads(Path(result.comparison_path).read_text(encoding="utf-8"))
            st.json(comparison_payload)
        except Exception:
            st.info("Comparison output was created, but it could not be rendered inline.")

    with st.expander("Execution Log", expanded=False):
        st.text(result.logs or "No log output captured.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_base_state()

    st.title(APP_TITLE)
    st.caption(
        "English-language research UI for configuring, tuning, and running "
        "config-driven DBP prediction experiments."
    )

    tabs = st.tabs(["Dataset", "Models", "Training", "Tuning", "Run"])

    preview_frame = None
    preview_error: str | None = None

    with tabs[0]:
        st.subheader("Dataset Setup")
        st.radio(
            "Data Source",
            options=["Use an existing file path", "Upload a dataset"],
            key="dataset_source_mode",
            horizontal=True,
        )

        if st.session_state["dataset_source_mode"] == "Upload a dataset":
            uploaded_file = st.file_uploader(
                "Upload CSV, Excel, or Parquet",
                type=["csv", "xlsx", "xls", "parquet"],
            )
            if uploaded_file is not None:
                saved_path = persist_uploaded_dataset(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                st.session_state["dataset_path"] = str(saved_path)
                suffix = saved_path.suffix.lower()
                if suffix in {".xlsx", ".xls"}:
                    st.session_state["dataset_format"] = "excel"
                elif suffix == ".parquet":
                    st.session_state["dataset_format"] = "parquet"
                else:
                    st.session_state["dataset_format"] = "csv"
                st.success(f"Uploaded dataset saved to `{saved_path}`.")

        data_cols = st.columns([2, 1])
        data_cols[0].text_input("Dataset Path", key="dataset_path")
        data_cols[1].selectbox(
            "File Format",
            options=["csv", "excel", "parquet"],
            key="dataset_format",
        )

        try:
            preview_frame = preview_dataset(
                st.session_state["dataset_path"],
                file_format=st.session_state["dataset_format"],
            )
            _sync_column_defaults(list(preview_frame.columns))
            st.markdown("**Preview**")
            st.dataframe(preview_frame, use_container_width=True)
            st.caption(
                f"{len(preview_frame.columns)} columns detected. "
                f"Showing up to {len(preview_frame)} preview rows."
            )
        except Exception as exc:
            preview_error = str(exc)
            st.error(preview_error)

        if preview_frame is not None:
            columns = list(preview_frame.columns)
            selection_cols = st.columns(2)
            selection_cols[0].multiselect(
                "Feature Columns",
                options=columns,
                key="feature_columns",
            )
            selection_cols[1].multiselect(
                "Target Columns",
                options=columns,
                key="target_columns",
            )

            target_cols = st.columns(2)
            target_cols[0].multiselect(
                "Prediction Targets",
                options=st.session_state["target_columns"],
                key="selected_targets",
                help="Subset of target columns to train in this run.",
            )
            target_cols[1].selectbox(
                "Split Column",
                options=columns,
                key="split_column",
            )

            label_cols = st.columns(2)
            label_cols[0].text_input("Train Label", key="train_label", value="train")
            label_cols[1].text_input("Test Label", key="test_label", value="test")

            with st.expander("Feature Pipeline (Advanced)", expanded=False):
                st.caption(
                    "Provide YAML or JSON for feature steps. Available transforms: "
                    + ", ".join(list_transform_names())
                )
                st.text_area(
                    "Feature Steps",
                    key="feature_steps_text",
                    height=180,
                    help="Example: [{'name': 'scale', 'params': {}}]",
                )

    with tabs[1]:
        st.subheader("Model Setup")
        st.caption("Choose which models to run and set their fixed base parameters.")
        for model_name in ordered_model_names():
            with st.expander(get_model_label(model_name), expanded=False):
                status_cols = st.columns([1, 2])
                status_cols[0].checkbox(
                    "Enable Model",
                    key=_key("model", model_name, "enabled"),
                )
                status_cols[1].text_input(
                    "Alias (optional)",
                    key=_key("model", model_name, "alias"),
                    help="Used for artifact naming when you want a custom label.",
                )

                field_specs = get_model_field_specs(model_name)
                if not field_specs:
                    st.info("No editable base parameters are defined for this model yet.")
                    continue

                grid_cols = st.columns(2)
                for index, field in enumerate(field_specs):
                    with grid_cols[index % 2]:
                        _render_field(field, _key("model", model_name, "params"))

    with tabs[2]:
        st.subheader("Training Defaults")
        st.caption("These settings act as the shared training baseline for all enabled models.")
        training_cols = st.columns(2)
        for index, field in enumerate(get_training_field_specs()):
            with training_cols[index % 2]:
                _render_field(field, "training")
        if st.session_state[_key("training", "loss")] != "Huber":
            st.info("`Huber Delta` is only used when the shared loss is set to `Huber`.")

    with tabs[3]:
        st.subheader("Hyperparameter Tuning")
        st.checkbox("Enable Tuning", key="tuning_enabled")
        if st.session_state["tuning_enabled"]:
            top_cols = st.columns([1.3, 1, 1, 1])
            top_cols[0].selectbox(
                "Tuning Preset",
                options=["Quick", "Standard", "Deep"],
                key="tuning_preset",
                on_change=_apply_tuning_preset,
            )
            top_cols[1].number_input("Trial Budget", min_value=1, step=1, key="global_trials")
            top_cols[2].number_input("CV Folds", min_value=2, step=1, key="global_folds")
            top_cols[3].number_input(
                "Stability Penalty",
                min_value=0.0,
                step=0.01,
                format="%.4f",
                key="global_stability_penalty",
            )
            st.checkbox(
                "Enable scout analysis in tuning outputs",
                key="tuning_scout",
                help="Writes additional Optuna analysis artifacts for post-run review.",
            )

            enabled_model_names = [
                model_name
                for model_name in ordered_model_names()
                if st.session_state[_key("model", model_name, "enabled")]
            ]
            if not enabled_model_names:
                st.warning("Enable at least one model to edit model-specific tuning settings.")

            for model_name in enabled_model_names:
                with st.expander(f"{get_model_label(model_name)} Tuning", expanded=False):
                    mode_cols = st.columns(2)
                    mode_cols[0].checkbox(
                        "Use Global Tuning Settings",
                        key=_key("model", model_name, "use_global_tuning"),
                    )
                    mode_cols[1].checkbox(
                        "Use Default Search Space",
                        key=_key("model", model_name, "use_default_search"),
                    )

                    if not st.session_state[_key("model", model_name, "use_global_tuning")]:
                        override_cols = st.columns(3)
                        override_cols[0].number_input(
                            "Model Trial Budget",
                            min_value=1,
                            step=1,
                            key=_key("model", model_name, "trials"),
                        )
                        override_cols[1].number_input(
                            "Model CV Folds",
                            min_value=2,
                            step=1,
                            key=_key("model", model_name, "folds"),
                        )
                        override_cols[2].number_input(
                            "Model Stability Penalty",
                            min_value=0.0,
                            step=0.01,
                            format="%.4f",
                            key=_key("model", model_name, "stability_penalty"),
                        )

                    if st.session_state[_key("model", model_name, "use_default_search")]:
                        st.info("The adapter-owned default search space will be used for this model.")
                    else:
                        _render_search_editor(model_name)

    with tabs[4]:
        st.subheader("Run Experiment")
        output_cols = st.columns([2, 1, 1])
        output_cols[0].text_input("Output Directory", key="output_dir")
        output_cols[1].checkbox("Save Models", key="save_models")
        output_cols[2].checkbox("Save Predictions", key="save_predictions")
        st.text_input(
            "Run ID (optional)",
            key="run_id",
            help="Leave blank to generate a timestamp-based run id.",
        )
        st.checkbox(
            "Inspect the dataset before execution",
            key="prepare_only_inspect_data",
            help="Creates a dataset snapshot and split summary before the run starts.",
        )

        payload: dict[str, Any] | None = None
        validation_error: str | None = preview_error

        if validation_error is None:
            try:
                feature_steps = parse_feature_steps(st.session_state["feature_steps_text"])
                training_params = _collect_field_values(get_training_field_specs(), "training")
                model_entries = _build_current_model_entries(
                    tuning_enabled=bool(st.session_state["tuning_enabled"])
                )
                payload = build_experiment_payload(
                    dataset_path=st.session_state["dataset_path"],
                    dataset_format=st.session_state["dataset_format"],
                    feature_columns=list(st.session_state.get("feature_columns", [])),
                    target_columns=list(st.session_state.get("target_columns", [])),
                    split_column=str(st.session_state.get("split_column", "")),
                    train_label=str(st.session_state.get("train_label", "train")),
                    test_label=str(st.session_state.get("test_label", "test")),
                    feature_steps=feature_steps,
                    selected_targets=list(st.session_state.get("selected_targets", [])),
                    model_entries=model_entries,
                    training_params=training_params,
                    tuning_enabled=bool(st.session_state["tuning_enabled"]),
                    tuning_params={
                        "trials": int(st.session_state["global_trials"]),
                        "folds": int(st.session_state["global_folds"]),
                        "stability_penalty": float(st.session_state["global_stability_penalty"]),
                        "scout": bool(st.session_state["tuning_scout"]),
                    },
                    output_dir=str(st.session_state["output_dir"]),
                    save_models=bool(st.session_state["save_models"]),
                    save_predictions=bool(st.session_state["save_predictions"]),
                )
            except Exception as exc:
                validation_error = str(exc)

        if payload is not None:
            workload = estimate_training_load(
                selected_models=payload["models"],
                target_count=len(payload["task"]["targets"]),
                tuning_enabled=bool(payload["tuning"]["enabled"]),
                global_trials=int(payload["tuning"]["trials"]),
                global_folds=int(payload["tuning"]["folds"]),
            )
            st.info(f"Estimated Training Cost: {workload['summary']}")

            preview_text = render_config_preview(payload)
            st.download_button(
                "Download Config",
                data=preview_text,
                file_name="dbp_experiment.yaml",
                mime="text/yaml",
            )
            st.code(preview_text, language="yaml")

            action_cols = st.columns(2)
            prepare_clicked = action_cols[0].button("Prepare Run", use_container_width=True)
            run_clicked = action_cols[1].button(
                "Run Experiment",
                type="primary",
                use_container_width=True,
            )

            if prepare_clicked or run_clicked:
                with st.spinner("Running the shared experiment engine..."):
                    result = execute_payload(
                        payload,
                        run_id=st.session_state["run_id"] or None,
                        prepare_only=prepare_clicked,
                        inspect_data=bool(st.session_state["prepare_only_inspect_data"]),
                    )
                st.session_state["last_execution"] = result
        elif validation_error is not None:
            st.error(validation_error)

        _render_execution_result()


if __name__ == "__main__":
    main()
