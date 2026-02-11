# Reads book.weeks.dat and generates docs/week1, etc

import mkdocs_gen_files
from pathlib import Path
import os
import tarfile
import io
from jinja2 import Environment, FileSystemLoader

jinja = Environment(
    loader=FileSystemLoader(os.getcwd()),
    autoescape=False,  # set True for HTML safety if needed
)

def gen_archive(week, name, files):
    fls = files.split()
    basedir = Path(fls[0])
    fls = fls[1:]

    f = io.BytesIO()
    with tarfile.open(fileobj=f, mode='w:gz') as tar:
        for filename in fls:
            print(f"  Adding {filename}")
            tar.add(str(basedir / Path(filename)), arcname=filename)

    with mkdocs_gen_files.open(f'week{week}/{name}', "wb") as out:
        out.write(f.getvalue())

def gen_jinja(week, name, files, extra_context={}):
    files = files.strip()
    print("JINJA", files)
    template = jinja.get_template(files)

    with mkdocs_gen_files.open(f'week{week}/{name}', 'wt') as out:
        out.write(template.render(**extra_context))

def do_gen(week, page, name, ty, files):
    print(f"Generating {name}")

    match ty.lower():
        case "archive":
            gen_archive(week, name, files)
        case "jinja":
            gen_jinja(week, name, files, {'page': page, 'week': week})

with open(os.environ.get("WEEKS_DAT", "Book/book.weeks.dat"), "rt") as f:
    week_counter = 0
    weeks = {}
    cur_week = None
    resources = {}

    def flush_week(w):
        cur_week = weeks[w]
        print(resources)
        items = sorted(list(resources.get(w, {}).items()), key=lambda x: x[0])
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
            cur_week_fp = f"week{week_counter}/index.md"
            weeks[week_counter] = cur_week = mkdocs_gen_files.open(cur_week_fp, "wt")
            cur_resources = resources.get(week_counter, {})
            resources[week_counter] = cur_resources
            print(f"# {name}\n", file=cur_week)
        if line.startswith('DESCRIPTION'):
            print(line.removeprefix("DESCRIPTION "), file=cur_week)
        if line.startswith('RESOURCE'):
            res = line.removeprefix('RESOURCE ')
            week, res = res.split(' ', 1)
            line, res = res.split(' ', 1)
            kind, res = res.split(' ', 1)

            week = int(week)
            ress = resources.get(week, {})
            resources[week] = ress
            ress[kind] = cur_resources.get(kind, []) + [(int(line), res)]
        if line.startswith('GEN'):
            res = line.removeprefix('GEN ')
            week, res = res.split(' ', 1)
            page, res = res.split(' ', 1)
            name, res = res.split(' ', 1)
            ty, res = res.split(' ', 1)

            do_gen(int(week), int(page), name, ty, res)

    for w in range(week_counter):
        flush_week(w + 1)

    nav = mkdocs_gen_files.Nav()
    nav["Home"] = "index.md"
    for i in range(week_counter):
        nav["Weeks", f"Week {i + 1}"] = f"week{i + 1}/index.md"
    nav["About"] = "about.md"
    with mkdocs_gen_files.open("SUMMARY.md", "wt") as f:
        print("FILE:" + '\n'.join(nav.build_literate_nav()))
        f.write(''.join(nav.build_literate_nav()))
