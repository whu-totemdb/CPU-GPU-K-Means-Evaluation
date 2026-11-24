"""Keep candidate-specific library paths exact; shared additions fill absent keys only."""
def execution_environment(original_function, parent, original, infrastructure):
    # In particular cuML registers its own LD_LIBRARY_PATH. The mlpack repair
    # must not prepend a foreign library prefix to that immutable candidate env.
    applicable = {key: value for key, value in infrastructure.items() if key not in original}
    return original_function(parent, original, applicable)

def is_cuml_binding_failure(row, job):
    return (job['algorithm'] == 'cuml_kmeans' and row.get('status') == 'MEASUREMENT_FAILED'
            and row.get('failure_stage') == 'binding'
            and row.get('error') == 'ValueError: EXECUTION_ENVIRONMENT_MISMATCH:LD_LIBRARY_PATH'
            and row.get('t_candidate_e2e') is None and row.get('t_algorithm_wall') is None
            and row.get('original_config_sha256') == job['original_config_sha256']
            and row.get('attempt_number', 1) == 1
            and not row.get('infrastructure_error'))
