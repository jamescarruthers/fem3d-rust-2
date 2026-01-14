"""
Note and Frequency Utilities

Provides conversion between musical note names and frequencies,
and utilities for generating note ranges.
"""

from typing import List, Optional, Literal
from dataclasses import dataclass
import math
import re


# Standard note names (sharps)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Note names with flats (for display)
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Natural notes only (white keys)
NATURAL_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Scale type
ScaleType = Literal['chromatic', 'natural', 'custom']


@dataclass
class NoteInfo:
    """Information about a musical note."""
    name: str           # e.g., "C4", "F#5"
    frequency: float    # Hz
    midi_number: int    # MIDI note number


def note_to_frequency(note_name: str) -> Optional[float]:
    """
    Convert a note name to frequency (A4 = 440 Hz).
    Accepts formats: "C4", "A#3", "Bb5", "F#2"
    """
    match = re.match(r'^([A-Ga-g])([#b]?)(\d)$', note_name)
    if not match:
        return None

    note, accidental, octave_str = match.groups()
    octave = int(octave_str)

    note_index = NOTE_NAMES.index(note.upper()) if note.upper() in NOTE_NAMES else -1
    if note_index == -1:
        return None

    if accidental == '#':
        note_index += 1
    if accidental == 'b':
        note_index -= 1

    # Handle wrap-around
    if note_index < 0:
        note_index += 12
    if note_index >= 12:
        note_index -= 12

    # Calculate semitones from A4 (A4 = 440 Hz, MIDI note 69)
    # A4 is note index 9 in octave 4
    semitones_from_a4 = (octave - 4) * 12 + (note_index - 9)

    # Frequency = 440 * 2^(semitones/12)
    return 440 * (2 ** (semitones_from_a4 / 12))


def frequency_to_note(freq: float) -> str:
    """Convert a frequency to the nearest note name."""
    # Convert frequency to nearest note name
    semitones_from_a4 = 12 * math.log2(freq / 440)
    midi_note = round(69 + semitones_from_a4)

    note_index = ((midi_note % 12) + 12) % 12
    octave = (midi_note // 12) - 1

    return f"{NOTE_NAMES[note_index]}{octave}"


def note_to_midi_number(note_name: str) -> Optional[int]:
    """
    Convert a note name to MIDI note number.
    C4 = 60, A4 = 69
    """
    match = re.match(r'^([A-Ga-g])([#b]?)(\d)$', note_name)
    if not match:
        return None

    note, accidental, octave_str = match.groups()
    octave = int(octave_str)

    note_index = NOTE_NAMES.index(note.upper()) if note.upper() in NOTE_NAMES else -1
    if note_index == -1:
        return None

    if accidental == '#':
        note_index += 1
    if accidental == 'b':
        note_index -= 1

    # Handle wrap-around
    if note_index < 0:
        note_index += 12
    if note_index >= 12:
        note_index -= 12

    # MIDI: C4 = 60, so octave 4 starts at 60
    return (octave + 1) * 12 + note_index


def midi_number_to_note(midi: int) -> str:
    """Convert MIDI note number to note name."""
    note_index = ((midi % 12) + 12) % 12
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


def generate_notes_in_range(
    start_note: str,
    end_note: str,
    scale_type: ScaleType,
    custom_notes: Optional[List[str]] = None
) -> List[NoteInfo]:
    """
    Generate a list of notes within a range.

    Args:
        start_note: Starting note (e.g., "F4")
        end_note: Ending note (e.g., "F5")
        scale_type: Type of scale: 'chromatic', 'natural', or 'custom'
        custom_notes: For custom scale type, list of note names to include

    Returns:
        Array of NoteInfo objects
    """
    start_midi = note_to_midi_number(start_note)
    end_midi = note_to_midi_number(end_note)

    if start_midi is None or end_midi is None:
        return []

    notes: List[NoteInfo] = []
    min_midi = min(start_midi, end_midi)
    max_midi = max(start_midi, end_midi)

    for midi in range(min_midi, max_midi + 1):
        note_index = ((midi % 12) + 12) % 12
        octave = (midi // 12) - 1
        note_name = NOTE_NAMES[note_index]
        full_name = f"{note_name}{octave}"

        # Filter based on scale type
        include = False

        if scale_type == 'chromatic':
            include = True
        elif scale_type == 'natural':
            include = note_name in NATURAL_NOTES
        elif scale_type == 'custom' and custom_notes:
            include = any(
                cn.upper().replace('B', '#').replace(' ', '') == full_name.upper()
                for cn in custom_notes
            )

        if include:
            freq = note_to_frequency(full_name)
            if freq is not None:
                notes.append(NoteInfo(
                    name=full_name,
                    frequency=freq,
                    midi_number=midi
                ))

    return notes


def generate_note_list() -> List[dict]:
    """Generate list of all notes for autocomplete (C2 to C7)."""
    notes: List[dict] = []
    for octave in range(2, 8):
        for note_name in NOTE_NAMES:
            note = f"{note_name}{octave}"
            freq = note_to_frequency(note)
            if freq and 20 <= freq <= 4000:
                notes.append({"note": note, "freq": round(freq, 1)})
    return notes


def format_frequency(freq: float) -> str:
    """Format a frequency with appropriate precision."""
    if freq >= 1000:
        return f"{freq:.1f}"
    elif freq >= 100:
        return f"{freq:.2f}"
    else:
        return f"{freq:.3f}"


def frequency_error_cents(computed: float, target: float) -> float:
    """
    Calculate error in cents between two frequencies.
    Positive = computed is sharp, Negative = computed is flat
    """
    if target <= 0 or computed <= 0:
        return 0.0
    return 1200 * math.log2(computed / target)
