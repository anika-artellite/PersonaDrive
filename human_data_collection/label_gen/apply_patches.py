#!/usr/bin/env python3
"""
Copy simlingo's language-label generators into this directory and patch them so
they run on our human-driven CARLA dataset:

    <root>/<persona>/<person>/<scenario>/<route>/Town12_Rep0_None/{boxes,measurements,rgb}/NNNN.*

instead of simlingo's own layout

    <root>/data/simlingo/<a>/<b>/<c>/<TownXX_RepN_..._routeM_ts>/{boxes,...}

Idempotent: always re-copies pristine sources from the repo, then patches.
Run:  python3 label_gen/apply_patches.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The bundled SimLingo tree is this package's sibling; $SIMLINGO_ROOT overrides it.
REPO = os.environ.get("SIMLINGO_ROOT") or os.path.join(os.path.dirname(HERE), "simlingo")

SRC = {
    "carla_vqa_generator.py": f"{REPO}/dataset_generation/language_labels/drivelm/carla_vqa_generator.py",
    "carla_vqa_generator_main.py": f"{REPO}/dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py",
    "carla_commentary_generator.py": f"{REPO}/dataset_generation/language_labels/commentary/carla_commentary_generator.py",
    "carla_commentary_generator_main.py": f"{REPO}/dataset_generation/language_labels/commentary/carla_commentary_generator_main.py",
}

n_applied = 0


def sub(text, old, new, label, count=1):
    global n_applied
    if old not in text:
        print(f"  !! PATCH NOT APPLIED ({label}): anchor not found", file=sys.stderr)
        sys.exit(1)
    occurrences = text.count(old)
    if occurrences != count:
        print(f"  !! PATCH AMBIGUOUS ({label}): {occurrences} occurrences, expected {count}",
              file=sys.stderr)
        sys.exit(1)
    n_applied += 1
    print(f"  ok  {label}")
    return text.replace(old, new)


# --- shared helper injected into both generators -----------------------------
HELPER = '''

# ----- patched-in helper: our layout ----------------------------------------
def _our_route_ids(path_measurements):
    """<root>/<persona>/<person>/<scenario>/<route>/<TownDir>/measurements/NNNN.json.gz"""
    p = path_measurements.split('/')
    return p[-5], p[-6], p[-4]   # scenario_name, route_file_number(person), route_number
# -----------------------------------------------------------------------------
'''


def patch_vqa_generator(t):
    # 1. glob: 5 wildcard levels, no data/simlingo prefix, drop the [:30] debug slice
    t = sub(t,
            "glob.glob(os.path.join(self.data_directory, 'data/simlingo/*/*/*/*/boxes/*.json.gz'))[:30]",
            "glob.glob(os.path.join(self.data_directory, '*/*/*/*/*/boxes/*.json.gz'))",
            "VQA glob depth 4->5, drop [:30]")

    # 2. force the expert-score route filter OFF (CLI cannot disable it: store_true default=True)
    t = sub(t,
            "        self.filter_routes_by_result = args.filter_routes_by_result\n",
            "        self.filter_routes_by_result = False  # PATCH: expert-only >=98 gate, wrong for human data\n",
            "VQA disable filter_routes_by_result")

    # 3. skip-existing output path: relpath instead of split('/data/')
    t = sub(t,
            '            save_dir = (self.output_directory + "/" + path.split("/data/")[1]).replace(\'boxes\', \'vqa\')',
            "            save_dir = os.path.join(self.output_directory, os.path.relpath(path, self.data_directory)).replace('/boxes/', '/vqa/')",
            "VQA skip-existing path")

    # 4. real output path: relpath instead of split('/data/')
    t = sub(t,
            '        save_dir = (self.output_directory + "/" + relative_image_path.split("/data/")[1]).replace(\'rgb\', \'vqa\').replace(\'.jpg\', \'.json.gz\')',
            "        save_dir = os.path.join(self.output_directory, os.path.relpath(relative_image_path, self.data_directory)).replace('/rgb/', '/vqa/').replace('.jpg', '.json.gz')",
            "VQA save path")

    # 5. scenario name from path components (get_scenario_name needs route XMLs we do not have)
    t = sub(t,
            """        else:
            try:
                scenario_name = get_scenario_name(path_measurements)
            except Exception as e:
                print(f"Error in {path_measurements}: {e}")
                return
            route_file_number = re.search(rf'Rep*(\\d+)_*(\\d+)_route_*(\\d+)', path_measurements).group(2)
            route_number = re.search(rf'Rep*(\\d+)_*(\\d+)_route_*(\\d+)', path_measurements).group(3)""",
            """        else:
            # PATCH: our <scenario> directory name IS the CARLA scenario type.
            scenario_name, route_file_number, route_number = _our_route_ids(path_measurements)""",
            "VQA scenario_name from path")

    # 6. remove the live breakpoint() and soften the assert (ParkingExit)
    t = sub(t,
            """                    if len(relevant_objects) > 1:
                        breakpoint()
                    assert len(relevant_objects) == 1, f'Expected 1 vehicle, found {len(relevant_objects)}, path: {self.current_path}'""",
            """                    if len(relevant_objects) != 1:
                        relevant_objects = relevant_objects[:1]  # PATCH: was breakpoint()+assert""",
            "VQA remove breakpoint/assert (ParkingExit)")

    # 7. static_trafficwarning boxes carry no 'type_id' -> KeyError (InvadingTurn)
    t = sub(t,
            "relevant_objects = list(filter(lambda x: x['type_id'] == 'static.prop.constructioncone'",
            "relevant_objects = list(filter(lambda x: x.get('type_id') == 'static.prop.constructioncone'",
            "VQA static type_id KeyError guard")

    return t + HELPER


def patch_vqa_main(t):
    # progress bar + loud stats on the serial path
    t = sub(t,
            """        for i in range(len_data_boxes):
            qas_generator.create_qa_pairs(i)""",
            """        import tqdm as _tqdm
        for i in _tqdm.tqdm(range(len_data_boxes)):
            qas_generator.create_qa_pairs(i)""",
            "VQA main progress bar")
    return t


def patch_commentary_generator(t):
    # 1. glob depth
    t = sub(t,
            "glob.glob(os.path.join(self.data_directory, 'data/simlingo/*/*/*/*/boxes/*.json.gz'))",
            "glob.glob(os.path.join(self.data_directory, '*/*/*/*/*/boxes/*.json.gz'))",
            "COM glob depth 4->5")

    # 2. force route filter off
    t = sub(t,
            "        self.filter_routes_by_result = args.filter_routes_by_result\n",
            "        self.filter_routes_by_result = False  # PATCH: expert-only >=98 gate, wrong for human data\n",
            "COM disable filter_routes_by_result")

    # 3. template file that ships under a different name
    t = sub(t,
            'template_file = f"data/augmented_templates/commentary.json"',
            'template_file = f"{os.environ.get(\'SIMLINGO_ROOT\', \'.\')}/data/augmented_templates/commentary_subsentence.json"',
            "COM template file -> commentary_subsentence.json")

    # 4. skip-existing output path
    t = sub(t,
            '            save_dir = (self.output_directory + "/" + path.split("/data/")[1]).replace(\'boxes\', \'commentary\')',
            "            save_dir = os.path.join(self.output_directory, os.path.relpath(path, self.data_directory)).replace('/boxes/', '/commentary/')",
            "COM skip-existing path")

    # 5. real output path
    t = sub(t,
            '        save_dir = (self.output_directory + "/" + path.split("/data/")[1]).replace(\'boxes\', \'commentary\')',
            "        save_dir = os.path.join(self.output_directory, os.path.relpath(path, self.data_directory)).replace('/boxes/', '/commentary/')",
            "COM save path")

    # 6. scenario name from path components (this call is NOT in a try/except upstream ->
    #    it kills the whole worker pool)
    t = sub(t,
            """        else:
            scenario_name = get_scenario_name(path_measurements)
            route_file_number = re.search(rf'Rep*(\\d+)_*(\\d+)_route_*(\\d+)', path_measurements).group(2)
            route_number = re.search(rf'Rep*(\\d+)_*(\\d+)_route_*(\\d+)', path_measurements).group(3)""",
            """        else:
            # PATCH: our <scenario> directory name IS the CARLA scenario type.
            scenario_name, route_file_number, route_number = _our_route_ids(path_measurements)""",
            "COM scenario_name from path")

    # 7. failed_paths.txt lands in a bogus directory once output != data dir
    t = sub(t,
            """            with open(f"{path.split('/data/')[0]}/commentary/failed_paths.txt", "a") as f:""",
            """            with open(f"{self.output_directory}/failed_paths.txt", "a") as f:""",
            "COM failed_paths.txt location")

    return t + HELPER


def patch_commentary_main(t):
    # chunksize 10000 with 64 workers leaves almost every worker idle
    t = sub(t,
            "r = process_map(com_generator.create_commentary, range(0, len_data_boxes), max_workers=64, chunksize=10000)",
            "_w = min(os.cpu_count() or 8, 24)\n        r = process_map(com_generator.create_commentary, range(0, len_data_boxes), max_workers=_w, chunksize=max(1, len_data_boxes // (_w * 8)))",
            "COM main worker/chunk sizing")
    if "\nimport os" not in t:
        t = sub(t, "import argparse", "import argparse\nimport os", "COM main import os")
    return t


PATCHERS = {
    "carla_vqa_generator.py": patch_vqa_generator,
    "carla_vqa_generator_main.py": patch_vqa_main,
    "carla_commentary_generator.py": patch_commentary_generator,
    "carla_commentary_generator_main.py": patch_commentary_main,
}

for name, src in SRC.items():
    print(f"{name}:")
    shutil.copy(src, os.path.join(HERE, name))
    with open(os.path.join(HERE, name)) as f:
        text = f.read()
    text = PATCHERS[name](text)
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)

print(f"\n{n_applied} patches applied into {HERE}")
