"""Utility functions for bar tuning optimization."""

from .note_utils import (
    NOTE_NAMES,
    NOTE_NAMES_FLAT,
    NATURAL_NOTES,
    note_to_frequency,
    frequency_to_note,
    note_to_midi_number,
    midi_number_to_note,
    generate_notes_in_range,
    generate_note_list,
    format_frequency,
    frequency_error_cents,
    NoteInfo,
    ScaleType,
)

from .bar_length_finder import (
    compute_f1_for_uniform_bar,
    find_optimal_length,
    find_lengths_for_notes,
    estimate_length_from_theory,
    LengthSearchResult,
)

__all__ = [
    # Note utils
    "NOTE_NAMES",
    "NOTE_NAMES_FLAT",
    "NATURAL_NOTES",
    "note_to_frequency",
    "frequency_to_note",
    "note_to_midi_number",
    "midi_number_to_note",
    "generate_notes_in_range",
    "generate_note_list",
    "format_frequency",
    "frequency_error_cents",
    "NoteInfo",
    "ScaleType",
    # Bar length finder
    "compute_f1_for_uniform_bar",
    "find_optimal_length",
    "find_lengths_for_notes",
    "estimate_length_from_theory",
    "LengthSearchResult",
]
