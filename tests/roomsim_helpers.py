"""Shared fixtures for the Room Sim tests (not a test module)."""

import json
import os

from wifisense.roomsim import parse_scene

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEDROOM = os.path.join(ROOT, "spec", "scenes", "bedroom.json")
HALL = os.path.join(ROOT, "spec", "scenes", "hall-l-shaped.json")
OFFICE = os.path.join(ROOT, "spec", "scenes", "office.json")
MULTI = os.path.join(ROOT, "spec", "scenes", "fixtures", "roomsim_multi.json")


def scene_dict(w=6.0, d=5.0, h=2.7, nodes=None, walls=None, objects=None, people=None, phone=None, polygon=None):
    room = {"w_m": w, "d_m": d, "h_m": h}
    if polygon is not None:
        room["polygon"] = polygon
    doc = {
        "schema_version": 1, "name": "test-room", "provenance": "hand_written", "room": room,
        "nodes": nodes if nodes is not None else [{"id": 0, "pos": [1.0, 2.5, 1.2]}, {"id": 1, "pos": [5.0, 2.5, 1.2]}],
    }
    if walls is not None:
        doc["walls"] = walls
    if objects is not None:
        doc["objects"] = objects
    if people is not None:
        doc["people"] = people
    if phone is not None:
        doc["phone"] = phone
    return doc


def make_scene(**kw):
    return parse_scene(json.dumps(scene_dict(**kw)))


def rect_walls(w, d, material="brick", t=0.23):
    pts = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
    return [{"from": list(pts[i]), "to": list(pts[(i + 1) % 4]), "material": material, "thickness_m": t}
            for i in range(4)]
