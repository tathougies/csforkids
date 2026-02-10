# Reads book.weeks.dat and generates docs/week1, etc

import mkdocs_gen_files
import os

with open(os.environ.get("WEEKS_DAT", "Book/book.weeks.dat"), "rt") as f:
    week_counter = 0
    cur_week = None
    cur_resources = {}

    def flush_week():
        items = sorted(list(cur_resources.items()), key=lambda x: x[0])
        for nm, subitems in items:
            cur_week.write(f"## {nm}\n\n")
            subitems.sort(key=lambda x: x[0])
            for i in subitems:
                cur_week.write(f"* {i[1]}")

        if cur_week is not None:
            cur_week.close()

    for line in f:
        if line.startswith('NAME'):
            week_counter += 1
            name = line.removeprefix("NAME ")
            flush_week()
            cur_week_fp = f"week{week_counter}/index.md"
            cur_week = mkdocs_gen_files.open(cur_week_fp, "wt")
            cur_resources = {}
            print(f"# {name}\n", file=cur_week)
        if line.startswith('DESCRIPTION'):
            print(line.removeprefix("DESCRIPTION "), file=cur_week)
        if line.startswith('RESOURCE'):
            res = line.removeprefix('RESOURCE ')
            line, res = res.split(' ', 1)
            kind, res = res.split(' ', 1)

            cur_resources[kind] = cur_resources.get(kind, []) + [(int(line), res)]
    flush_week()
    nav = mkdocs_gen_files.Nav()
    nav["Home"] = "index.md"
    for i in range(week_counter):
        nav["Weeks", f"Week {i + 1}"] = f"week{i + 1}/index.md"
    nav["About"] = "about.md"
    with mkdocs_gen_files.open("SUMMARY.md", "wt") as f:
        print("FILE:" + '\n'.join(nav.build_literate_nav()))
        f.write(''.join(nav.build_literate_nav()))
