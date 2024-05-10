from .tree import Tree

def validate_conllu(*conllu_file_paths: str):
    errors: dict[str, list[AssertionError]] = {conllu_file_path: [] for conllu_file_path in conllu_file_paths}
    for conllu_file_path, error_list in errors.items():
        print(f"Validating {conllu_file_path}... ", end='', flush=True)
        with open(conllu_file_path, encoding="utf-8") as conllu_file:
            for raw_conllu in conllu_file.read().split("\n\n"):
                if raw_conllu != '':
                    try:
                        Tree(raw_conllu)
                    except AssertionError as e:
                        error_list.append(e)
        if error_list:
            print("Failed")
        else:
            print("Passed")
    if any(errors.values()):
        print("\nReasons for Failure (only the first one encountered in each tree):")
        for i, (conllu_file_path, error_list) in enumerate(errors.items(), start=1):
            if error_list:
                print(f"{i}) {conllu_file_path}:\n" + '\n'.join(str(error) for error in error_list))
