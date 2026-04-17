"""Rescale humanoid.urdf: s=1/3.25, target mass ~70kg, fix wrists, add head. Not run at runtime."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

S = 1.0 / 3.25
OLDSUM = 93.0
NEWMASS = 70.0
MF = NEWMASS / OLDSUM
INF = MF * S * S


def sf(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def scale_xyz(attr: str | None) -> str | None:
    if not attr:
        return attr
    parts = attr.replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(sf(float(p) * S))
        except ValueError:
            out.append(p)
    return " ".join(out)


def process_origin(el: ET.Element) -> None:
    if el.get("xyz"):
        el.set("xyz", scale_xyz(el.get("xyz")) or "0 0 0")
    # rpy: angles, do not scale


def process_geometry(geom: ET.Element) -> None:
    for g in list(geom):
        tag = g.tag.split("}")[-1] if "}" in g.tag else g.tag
        if tag == "sphere" and g.get("radius"):
            g.set("radius", sf(float(g.get("radius", "0")) * S))
        elif tag == "box" and g.get("size"):
            a, b, c = map(float, g.get("size", "0 0 0").split())
            g.set("size", f"{sf(a*S)} {sf(b*S)} {sf(c*S)}")
        elif tag == "capsule":
            if g.get("length"):
                g.set("length", sf(float(g.get("length", "0")) * S))
            if g.get("radius"):
                g.set("radius", sf(float(g.get("radius", "0")) * S))


def main() -> None:
    p = Path(__file__).with_name("humanoid.urdf")
    tree = ET.parse(p)
    root = tree.getroot()

    for link in root.findall(".//link"):
        name = link.get("name", "")
        for inertial in link.findall("inertial"):
            for o in inertial.findall("origin"):
                process_origin(o)
            mass_el = inertial.find("mass")
            if mass_el is not None and mass_el.get("value"):
                mv = float(mass_el.get("value", "0"))
                if mv > 0.001:
                    mass_el.set("value", sf(mv * MF))
            inertia_el = inertial.find("inertia")
            if inertia_el is not None and name != "base":
                for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    if inertia_el.get(k) is not None:
                        inertia_el.set(k, sf(float(inertia_el.get(k, "0")) * INF))
        for col in link.findall("collision"):
            for o in col.findall("origin"):
                process_origin(o)
            geom = col.find("geometry")
            if geom is not None:
                process_geometry(geom)
        for vis in link.findall("visual"):
            for o in vis.findall("origin"):
                process_origin(o)
            geom = vis.find("geometry")
            if geom is not None:
                process_geometry(geom)

    for joint in root.findall(".//joint"):
        for o in joint.findall("origin"):
            process_origin(o)
    # Revolute wrists (origins already scaled above); left had invalid XML before rescale
    for jname in ("right_wrist", "left_wrist"):
        j = root.find(f'.//joint[@name="{jname}"]')
        if j is None:
            continue
        j.set("type", "revolute")
        o = j.find("origin")
        if o is None:
            o = ET.SubElement(j, "origin")
        o.set("rpy", "0 0 0")
        if jname == "left_wrist":
            o.set("xyz", f"0 0 -{sf(1.035788 * S)}")
        axis = j.find("axis")
        if axis is None:
            axis = ET.SubElement(j, "axis")
        axis.set("xyz", "0 0 1")
        old_lim = j.find("limit")
        if old_lim is not None:
            j.remove(old_lim)
        lim = ET.SubElement(j, "limit")
        lim.set("effort", "200")
        lim.set("lower", "-0.8")
        lim.set("upper", "0.8")
        lim.set("velocity", "2")

    # Add head (fixed to neck); neck link name is "neck"
    neck_link = root.find('.//link[@name="neck"]')
    neck_j = root.find('.//joint[@name="neck"]')
    if neck_link is not None and neck_j is not None:
        # Offset along neck +Y to top of scaled neck sphere (com0.7*s + radius 0.41*s)
        top_y = (0.7 + 0.41) * S
        head = ET.Element("link", {"name": "head"})
        head_in = ET.SubElement(head, "inertial")
        ho = ET.SubElement(head_in, "origin", {"rpy": "0 0 0", "xyz": f"0 {sf(0.06)} 0"})
        ET.SubElement(head_in, "mass", {"value": sf(4.5 * MF)})
        ET.SubElement(
            head_in,
            "inertia",
            {
                "ixx": sf(0.03 * INF),
                "ixy": "0",
                "ixz": "0",
                "iyy": sf(0.03 * INF),
                "iyz": "0",
                "izz": sf(0.03 * INF),
            },
        )
        hcol = ET.SubElement(head, "collision")
        ET.SubElement(hcol, "origin", {"rpy": "0 0 0", "xyz": f"0 {sf(0.06)} 0"})
        hg = ET.SubElement(hcol, "geometry")
        ET.SubElement(hg, "sphere", {"radius": sf(0.12 * S)})
        hvis = ET.SubElement(head, "visual")
        ET.SubElement(hvis, "origin", {"rpy": "0 0 0", "xyz": f"0 {sf(0.06)} 0"})
        hvg = ET.SubElement(hvis, "geometry")
        ET.SubElement(hvg, "sphere", {"radius": sf(0.12 * S)})

        hj = ET.Element("joint", {"name": "head", "type": "fixed"})
        ET.SubElement(hj, "parent", {"link": "neck"})
        ET.SubElement(hj, "child", {"link": "head"})
        ET.SubElement(
            hj,
            "origin",
            {"rpy": "0 0 0", "xyz": f"0 {sf(top_y)} 0"},
        )

        def _is_link(el: ET.Element) -> bool:
            t = el.tag.split("}")[-1]
            return t == "link"

        def _is_joint(el: ET.Element) -> bool:
            t = el.tag.split("}")[-1]
            return t == "joint"

        ch = list(root)
        ni = next(i for i, c in enumerate(ch) if _is_link(c) and c.get("name") == "neck")
        root.insert(ni + 1, head)
        ch = list(root)
        ji = next(i for i, c in enumerate(ch) if _is_joint(c) and c.get("name") == "neck")
        root.insert(ji + 1, hj)

    ET.indent(tree, space="\t")
    tree.write(p, encoding="utf-8", xml_declaration=False)
    print("OK:", p)


if __name__ == "__main__":
    main()
