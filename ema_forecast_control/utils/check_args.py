def check_args(args: dict) -> dict:

    args = check_input_features(args)
    return args

def check_input_features(args: dict) -> dict:
    if isinstance(args['input_features'], list) and len(args['input_features']) == 0:
        args['input_features'] = None
    return args