import sys
import csv
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QListView,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QDialog,
    QSlider,
    QFormLayout,
    QDialogButtonBox,
    QSizePolicy,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QInputDialog,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont
# Matplotlib for plotting in dialogs
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import re
import math


def _maybe_force_xcb_on_wayland():
    """If running under Wayland, restart the process with QT_QPA_PLATFORM=xcb.

    This avoids QWindow::requestActivate() limitations on some Wayland compositors.
    """
    try:
        if 'WAYLAND_DISPLAY' in os.environ and not os.environ.get('QT_QPA_PLATFORM'):
            os.environ['QT_QPA_PLATFORM'] = 'xcb'
            python = sys.executable
            args = [python] + sys.argv
            print("Detected Wayland session — restarting with QT_QPA_PLATFORM=xcb for compatibility...")
            os.execv(python, args)
    except Exception:
        pass


UI_SCALE = 1.25


def _scale_px(value: int) -> int:
    return int(round(value * UI_SCALE))


def apply_ui_scale(app: QApplication, scale: float = UI_SCALE) -> None:
    base = app.font()
    scaled = QFont(base)
    if base.pointSizeF() and base.pointSizeF() > 0:
        scaled.setPointSizeF(base.pointSizeF() * scale)
    elif base.pixelSize() and base.pixelSize() > 0:
        scaled.setPixelSize(int(round(base.pixelSize() * scale)))
    app.setFont(scaled)


def read_criteria(path: Path):
    criteria = []
    if not path.exists():
        return criteria
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get('name') or r.get('indicator') or r.get('criterion') or r.get('id')
            group = r.get('group') or r.get('group_name') or r.get('G') or ''
            # try common min/max field names
            def parse_num(keys, default):
                for k in keys:
                    v = r.get(k)
                    if v is not None and v != '':
                        try:
                            return float(v)
                        except Exception:
                            pass
                return default

            lo = parse_num(['min', 'lower', 'min_value', 'low'], 0.0)
            hi = parse_num(['max', 'upper', 'max_value', 'high'], 1.0)
            if name is None:
                # try first column value
                name = list(r.values())[0]
            criteria.append({'name': name, 'min': lo, 'max': hi, 'group': group})
    return criteria


def read_alternatives(path: Path):
    # flexible reader based on the user's described format
    if not path.exists():
        return [], []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return [], []
        # If first row looks like header with alternative names in columns 1..
        header = rows[0]
        # If a 'confidence' column is present, remove that column from header and all rows
        conf_indices = [i for i, h in enumerate(header) if (h or '').strip().lower() == 'confidence']
        if conf_indices:
            # remove those indices from each row
            new_rows = []
            for r in rows:
                new_r = [v for i, v in enumerate(r) if i not in conf_indices]
                new_rows.append(new_r)
            rows = new_rows
            header = rows[0]
        # If header first cell is 'indicator' then this is the qualitative format
        exclude_headers = {'meta', 'metadata', 'notes', 'note', 'comment', 'info'}
        if header[0].strip().lower() == 'indicator':
            alt_names = [h.strip() for h in header[1:] if h.strip() and h.strip().lower() not in exclude_headers]
            # collect rows that are "- starting point" entries
            start_entries = []
            for row in rows[1:]:
                key = row[0].strip()
                if key.lower().endswith('starting point') or 'starting point' in key.lower():
                    # indicator name is the part before ' - starting point' if present
                    base = key.split('-')[0].strip()
                    values = [c for c in row[1:len(alt_names)+1]]
                    # try parse numeric positions
                    parsed = []
                    for v in values:
                        try:
                            parsed.append(float(v))
                        except Exception:
                            parsed.append(v)
                    # Ignore any extra/meta columns; only collect start points for alternatives
                    start_entries.append({'indicator': base, 'raw_label': key, 'start_points': parsed})
            return alt_names, start_entries
        else:
            # otherwise assume first row contains alternative names across the row
            alt_names = [c.strip() for c in header if c.strip() and c.strip().lower() not in exclude_headers]
            # second row as starting points
            if len(rows) > 1:
                try:
                    start_points = [float(x) for x in rows[1][: len(alt_names)]]
                except Exception:
                    start_points = []
            else:
                start_points = []
            return alt_names, start_points


class RankingWindow(QWidget):
    def __init__(self, alt_names, start_points, indicator_name=None, mode='ranking'):
        super().__init__()
        self.setWindowTitle('Ranking - drag to reorder; select multiple to tie')
        # sensible default size so contents are visible on open
        self.resize(_scale_px(700), _scale_px(480))
        self.setMinimumSize(_scale_px(600), _scale_px(360))
        self.alt_names = alt_names
        self.start_points = start_points
        self.indicator_name = indicator_name

        layout = QVBoxLayout()

        # Header instructions moved to the main status label; don't add top labels here
        header_text = 'Reorder alternatives by dragging. Select multiple and click "Tie Selected" to place them at the same rank.'
        if self.indicator_name:
            header_text = f"Indicator: {self.indicator_name} — " + header_text

        self.listw = QListWidget()
        self.listw.setDragDropMode(QListWidget.InternalMove)
        # allow selecting multiple items (Shift/Ctrl click) so user can tie them
        self.listw.setSelectionMode(QListWidget.ExtendedSelection)
        # expand to fill available space so items are visible
        self.listw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.listw)
        # push the controls (buttons / confidence radios) to the bottom
        layout.addStretch(1)

        # Create a bottom panel that will host confidence radios above the action buttons
        self.bottom_panel = QWidget()
        bp_layout = QVBoxLayout()
        bp_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_panel.setLayout(bp_layout)

        # inner horizontal layout for action buttons (will sit below the confidence radios)
        buttons_h = QHBoxLayout()
        # mode toggle: ranking (vertical) vs spectrum (horizontal)
        self.mode_btn = QPushButton('Switch to Spectrum')
        buttons_h.addWidget(self.mode_btn)
        self.mode_btn.clicked.connect(self.toggle_mode)
        self.mode = mode

        # country-dependent helpers: allow copying from another country entry
        self.copy_from_btn = QPushButton('Copy from...')
        self.copy_from_btn.clicked.connect(self.on_copy_from)
        # only show for labels that look like "Base - CC"
        try:
            show_copy = bool(indicator_name and (' - ' in str(indicator_name)) and re.match(r'.+\s-\s[A-Za-z]{2,3}$', str(indicator_name).strip()))
        except Exception:
            show_copy = False
        self.copy_from_btn.setVisible(show_copy)
        self.copy_from_btn.setEnabled(show_copy)
        buttons_h.addWidget(self.copy_from_btn)

        tie_btn = QPushButton('Tie Selected')
        untie_btn = QPushButton('Untie Selected')
        up_btn = QPushButton('Move Up')
        down_btn = QPushButton('Move Down')
        buttons_h.addWidget(tie_btn)
        buttons_h.addWidget(untie_btn)
        buttons_h.addWidget(up_btn)
        buttons_h.addWidget(down_btn)
        # Navigation is handled by MainApp's persistent Back/Next buttons.
        # add buttons_h to the vertical bottom panel (confidence radios will be inserted above later)
        bp_layout.addLayout(buttons_h)

        tie_btn.clicked.connect(self.tie_selected)
        untie_btn.clicked.connect(self.untie_selected)
        up_btn.clicked.connect(self.move_up)
        down_btn.clicked.connect(self.move_down)

        # store refs for mode-dependent enabling/disabling
        self._tie_btn = tie_btn
        self._untie_btn = untie_btn
        self._up_btn = up_btn
        self._down_btn = down_btn

        self.setLayout(layout)
        self.populate()

        # Confidence selector for this indicator elicitation (0..4, default 3)
        try:
            conf_layout = QHBoxLayout()
            conf_layout.addWidget(QLabel('Confidence:'))
            self._conf_group = QButtonGroup(self)
            self._conf_buttons = []
            for i in range(5):
                rb = QRadioButton(str(i))
                conf_layout.addWidget(rb)
                self._conf_group.addButton(rb, i)
                self._conf_buttons.append(rb)
            # default to 3
            try:
                self._conf_buttons[3].setChecked(True)
            except Exception:
                self._conf_buttons[2].setChecked(True)
            # place confidence controls into the bottom panel above the action buttons
            try:
                bp_layout.insertLayout(0, conf_layout)
            except Exception:
                bp_layout.addLayout(conf_layout)
        except Exception:
            # non-critical; continue without confidence selector
            self._conf_group = None

    def on_copy_from(self):
        try:
            top = self.window()
            if hasattr(top, 'copy_from_for_current_indicator'):
                top.copy_from_for_current_indicator(kind='ranking')
        except Exception:
            pass

    def apply_start_points(self, start_points):
        try:
            self.start_points = start_points
            self.populate()
        except Exception:
            pass

    def populate(self):
        self.listw.clear()
        items = list(self.alt_names)
        # if starting points are provided, sort by them to create initial ranking
        # and collapse equal scores into tied items (comma-separated labels)
        if self.start_points and len(self.start_points) >= len(items):
            try:
                paired = list(zip(items, self.start_points))
                # convert scores to floats where possible; use -inf for non-numeric so they sort last
                def _safe_float(v):
                    try:
                        return float(v)
                    except Exception:
                        return float('-inf')
                paired = [(a, _safe_float(s)) for a, s in paired]
                # scores in the alternatives file use higher == better (P = best).
                # Sort descending so best items appear first in the ranking/spectrum.
                paired.sort(key=lambda p: p[1], reverse=True)
                # group consecutive items with equal scores into tied composite items
                grouped = []
                i = 0
                while i < len(paired):
                    a, score = paired[i]
                    group_names = [a]
                    j = i + 1
                    while j < len(paired) and math.isclose(paired[j][1], score, rel_tol=1e-9, abs_tol=1e-12):
                        group_names.append(paired[j][0])
                        j += 1
                    if len(group_names) == 1:
                        grouped.append(group_names[0])
                    else:
                        # tie: join names with comma
                        grouped.append(','.join(group_names))
                    i = j
                items = grouped
            except Exception:
                pass
        for a in items:
            it = QListWidgetItem(a)
            if self.mode == 'spectrum':
                # center text for spectrum items
                try:
                    it.setTextAlignment(Qt.AlignCenter)
                except Exception:
                    pass
                # set a uniform size for spectrum items so ties (comma-joined labels)
                # don't create wildly different item widths and spacing
                try:
                    it.setSizeHint(QSize(140, 48))
                except Exception:
                    pass
            self.listw.addItem(it)
        # apply mode-specific list settings
        try:
            if self.mode == 'spectrum':
                # horizontal left-to-right flow without IconMode/grid (avoids large gaps)
                self.listw.setFlow(QListView.LeftToRight)
                self.listw.setWrapping(False)
                try:
                    # Use ListMode with Snap movement so dragging reorders left-to-right predictably
                    self.listw.setViewMode(QListView.ListMode)
                    self.listw.setMovement(QListView.Snap)
                    # small spacing so items appear close together in a row
                    self.listw.setSpacing(6)
                    # use uniform item sizes to avoid layout jumps when labels change
                    try:
                        self.listw.setUniformItemSizes(True)
                    except Exception:
                        pass
                except Exception:
                    pass
                # allow tie/untie also in spectrum mode
                self._tie_btn.setEnabled(True)
                self._untie_btn.setEnabled(True)
            else:
                # vertical top-to-bottom; restore default movement/spacing
                self.listw.setFlow(QListView.TopToBottom)
                try:
                    self.listw.setViewMode(QListView.ListMode)
                    self.listw.setMovement(QListView.Free)
                    self.listw.setSpacing(0)
                    try:
                        self.listw.setUniformItemSizes(False)
                    except Exception:
                        pass
                except Exception:
                    pass
                self._tie_btn.setEnabled(True)
                self._untie_btn.setEnabled(True)
        except Exception:
            pass

    def tie_selected(self):
        sels = self.listw.selectedItems()
        if len(sels) <= 1:
            return
        texts = [s.text() for s in sels]
        first_row = self.listw.row(sels[0])
        # remove items (from last to first to keep indices valid)
        for s in sorted(sels, key=lambda it: self.listw.row(it), reverse=True):
            self.listw.takeItem(self.listw.row(s))
        composite = ','.join(texts)
        item = QListWidgetItem(composite)
        self.listw.insertItem(first_row, item)

    def untie_selected(self):
        sels = self.listw.selectedItems()
        if not sels:
            return
        # operate on a snapshot to avoid mutation during iteration
        for s in list(sels):
            txt = s.text()
            if ',' in txt:
                row = self.listw.row(s)
                self.listw.takeItem(row)
                parts = [p.strip() for p in txt.split(',') if p.strip()]
                for i, p in enumerate(parts):
                    self.listw.insertItem(row + i, QListWidgetItem(p))

    def move_up(self):
        sels = self.listw.selectedItems()
        if not sels:
            return
        for s in sels:
            row = self.listw.row(s)
            if row > 0:
                it = self.listw.takeItem(row)
                self.listw.insertItem(row - 1, it)
                it.setSelected(True)

    def move_down(self):
        sels = list(self.listw.selectedItems())
        if not sels:
            return
        # move down starting from bottom
        for s in sorted(sels, key=lambda it: self.listw.row(it), reverse=True):
            row = self.listw.row(s)
            if row < self.listw.count() - 1:
                it = self.listw.takeItem(row)
                self.listw.insertItem(row + 1, it)
                it.setSelected(True)

    def get_ranks(self):
        ranks = []
        for i in range(self.listw.count()):
            txt = self.listw.item(i).text()
            parts = [p.strip() for p in txt.split(',') if p.strip()]
            ranks.append(parts)
        return ranks

    def get_confidence(self):
        try:
            return self._conf_group.checkedId() if self._conf_group is not None else None
        except Exception:
            return None

    def toggle_mode(self):
        # switch between 'ranking' and 'spectrum'
        try:
            if self.mode == 'ranking':
                self.mode = 'spectrum'
                self.mode_btn.setText('Switch to Ranking')
                # give user feedback via parent status if available
                try:
                    top = self.window()
                    if hasattr(top, 'set_status'):
                        top.set_status('Spectrum mode: order alternatives left-to-right')
                except Exception:
                    pass
            else:
                self.mode = 'ranking'
                self.mode_btn.setText('Switch to Spectrum')
                try:
                    top = self.window()
                    if hasattr(top, 'set_status'):
                        top.set_status('Ranking mode: order alternatives top-to-bottom')
                except Exception:
                    pass
            # re-populate to apply new flow and button states
            self.populate()
        except Exception:
            pass


class ValueFunctionWidget(QWidget):
    def __init__(self, criterion, points_count, xs_override=None, ys_override=None, point_labels=None, on_save=None, on_cancel=None, parent=None):
        super().__init__(parent)
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.setWindowTitle(f"Value function - {criterion['name']}")
        # make widget large enough to show sliders clearly
        self.resize(700, 420)
        self.setMinimumSize(560, 360)
        self.criterion = criterion
        self.points_count = points_count
        self.xs = []
        self.ys = []
        # endpoints: first and last points are treated as fixed dummy endpoints
        # controlled via toggles (0 or 1). Callers should include endpoints
        # in points_count when they want them present.
        self.has_endpoints = (self.points_count >= 2)
        # allow caller to specify explicit X positions (e.g., ranks 1..P for qualitative indicators)
        if xs_override is not None:
            try:
                # ensure floats
                self.xs = [float(x) for x in xs_override]
            except Exception:
                self.xs = [float(x) for x in xs_override]
        else:
            lo = criterion.get('min', 0.0)
            hi = criterion.get('max', 1.0)
            if points_count == 1:
                self.xs = [(lo + hi) / 2.0]
            else:
                self.xs = [lo + i * (hi - lo) / (points_count - 1) for i in range(points_count)]
        # default linear increasing, but allow caller to override Y positions
        if ys_override is not None:
            try:
                # ensure floats and match length
                self.ys = [float(y) for x, y in ys_override] if all(isinstance(t, (list, tuple)) and len(t) == 2 for t in ys_override) else [float(y) for y in ys_override]
                # if provided series length mismatches points_count, adapt by trimming or padding
                if len(self.ys) < points_count:
                    # pad with last value
                    last = self.ys[-1] if self.ys else 1.0
                    self.ys += [last] * (points_count - len(self.ys))
                elif len(self.ys) > points_count:
                    self.ys = self.ys[:points_count]
            except Exception:
                if points_count == 1:
                    self.ys = [1.0]
                else:
                    self.ys = [i / (points_count - 1) for i in range(points_count)]
        else:
            if points_count == 1:
                self.ys = [1.0]
            else:
                # default linear increasing between 0 and 1
                self.ys = [i / (points_count - 1) for i in range(points_count)]

        layout = QVBoxLayout()
        form = QFormLayout()
        self.sliders = []
        self.value_labels = []
        # endpoint toggles (only meaningful when there are at least two points)
        if self.has_endpoints:
            self.left_toggle = QCheckBox('Left endpoint = 1')
            self.right_toggle = QCheckBox('Right endpoint = 1')
            # set initial toggle states based on initial ys if available
            try:
                self.left_toggle.setChecked(bool(round(self.ys[0]) == 1))
                self.right_toggle.setChecked(bool(round(self.ys[-1]) == 1))
            except Exception:
                self.left_toggle.setChecked(False)
                self.right_toggle.setChecked(True)
            htog = QHBoxLayout()
            htog.addWidget(self.left_toggle)
            htog.addStretch(1)
            htog.addWidget(self.right_toggle)
            form.addRow(QLabel('Endpoints:'), htog)
        # allow custom labels for each point (e.g., the alternative names at that rank)
        if point_labels is None:
            point_labels = [None] * points_count
        for i in range(points_count):
            pl = point_labels[i] if i < len(point_labels) else None
            if pl:
                label = QLabel(f"{pl} (x={self.xs[i]:.3g})")
            else:
                label = QLabel(f"Point {i+1} (x={self.xs[i]:.3g})")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(self.ys[i] * 100))
            # endpoints are controlled by toggles and locked (disabled)
            if self.has_endpoints and (i == 0 or i == points_count - 1):
                # hide sliders for endpoints while keeping them in the logic
                try:
                    slider.setVisible(False)
                except Exception:
                    try:
                        slider.hide()
                    except Exception:
                        pass
            # otherwise editable
            self.sliders.append(slider)
            val_label = QLabel(f"{self.ys[i]:.2f}")
            self.value_labels.append(val_label)
            h = QHBoxLayout()
            h.addWidget(slider)
            h.addWidget(val_label)
            # hide the numeric label for endpoints as well to reduce visual clutter
            if self.has_endpoints and (i == 0 or i == points_count - 1):
                try:
                    val_label.setVisible(False)
                except Exception:
                    try:
                        val_label.hide()
                    except Exception:
                        pass
            form.addRow(label, h)
        # wrap the form into a widget so we can control vertical stretch separately
        slider_container = QWidget()
        slider_container.setLayout(form)
        # add slider area with minimal stretch so the plot can take most vertical space
        layout.addWidget(slider_container, 0)

        # Add a matplotlib canvas to show the current value function
        # Use a shorter figure height but give the canvas more layout stretch
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        # ensure the canvas expands to fill the container and is drawn immediately
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # add canvas with higher stretch factor so it receives more vertical space
        layout.addWidget(self.canvas, 5)
        self._plot_line, = self.ax.plot(self.xs, self.ys, marker='o')
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True)
        try:
            self.ax.set_xlim(min(self.xs), max(self.xs))
        except Exception:
            pass
        self.ax.set_ylabel('Value')
        self.ax.set_xlabel('Criterion')
        # store labels for each plotted X so ticks can show alternative names
        try:
            self.point_labels = point_labels if point_labels is not None else [None] * len(self.xs)
        except Exception:
            self.point_labels = [None] * len(self.xs)

        # set tick labels using provided point labels (replace commas with newline)
        try:
            pairs = list(zip(self.xs, self.point_labels))
            pairs_sorted = sorted(pairs, key=lambda t: float(t[0]))
            xs_sorted, labels_sorted = zip(*pairs_sorted)
            proc_labels = [ (str(l).replace(',', '\n') if l is not None else '') for l in labels_sorted ]
            self.ax.set_xticks(xs_sorted)
            self.ax.set_xticklabels(proc_labels, rotation=0, ha='center')
        except Exception:
            pass

        self.fig.tight_layout()
        # Connect sliders to update plot and value labels
        for idx, s in enumerate(self.sliders):
            s.valueChanged.connect(lambda _v, i=idx: self.on_slider_changed(i))

        # connect endpoint toggles to update endpoint slider values
        if self.has_endpoints:
            try:
                self.left_toggle.toggled.connect(lambda v: self._apply_endpoint_toggle(0, v))
                self.right_toggle.toggled.connect(lambda v: self._apply_endpoint_toggle(self.points_count - 1, v))
            except Exception:
                pass

        # draw initial plot so widget is not blank when shown
        self.update_plot()
        try:
            self.canvas.draw()
        except Exception:
            pass

        # ensure the plot/canvas expands; bottom controls will be placed in MainApp.bottom_bar

        # Build a bottom panel containing the VF confidence selector and action buttons.
        # This panel will be reparented into MainApp's bottom bar when the view is shown.
        self.bottom_panel = QWidget()
        bp_layout = QVBoxLayout()
        bp_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_panel.setLayout(bp_layout)

        # Confidence selector for this value function (0..4, default 3)
        try:
            conf_row = QHBoxLayout()
            conf_row.addWidget(QLabel('VF Confidence:'))
            self._vf_conf_group = QButtonGroup(self)
            self._vf_conf_buttons = []
            for i in range(5):
                rb = QRadioButton(str(i))
                conf_row.addWidget(rb)
                self._vf_conf_group.addButton(rb, i)
                self._vf_conf_buttons.append(rb)
            try:
                self._vf_conf_buttons[3].setChecked(True)
            except Exception:
                self._vf_conf_buttons[2].setChecked(True)
            bp_layout.addLayout(conf_row)
        except Exception:
            self._vf_conf_group = None

        btns = QHBoxLayout()
        # Shape cycle button: cycles between Increasing and Decreasing
        self.shape_states = ['Linear Increasing', 'Linear Decreasing']
        self.shape_index = 0
        self.shape_btn = QPushButton(self.shape_states[self.shape_index])
        btns.addWidget(self.shape_btn)

        # country-dependent helpers: allow copying from another country entry
        self.copy_from_btn = QPushButton('Copy from...')
        self.copy_from_btn.clicked.connect(self.on_copy_from)
        try:
            nm = str(self.criterion.get('name', '')).strip()
            show_copy = bool(nm and (' - ' in nm) and re.match(r'.+\s-\s[A-Za-z]{2,3}$', nm))
        except Exception:
            show_copy = False
        self.copy_from_btn.setVisible(show_copy)
        self.copy_from_btn.setEnabled(show_copy)
        btns.addWidget(self.copy_from_btn)
        # Navigation is handled by MainApp's persistent Back/Next buttons.
        bp_layout.addLayout(btns)
        self.shape_btn.clicked.connect(self.cycle_shape)
        

        # do not add bottom_panel to the main layout here; MainApp will place it in its bottom bar
        self.setLayout(layout)

    def on_copy_from(self):
        try:
            top = self.window()
            if hasattr(top, 'copy_from_for_current_indicator'):
                top.copy_from_for_current_indicator(kind='value_function')
        except Exception:
            pass

    def apply_points(self, points, confidence=None):
        """Apply a set of (x,y) points into the sliders, and optionally set VF confidence."""
        if not points:
            return
        try:
            ys = []
            # points may be [(x,y),...] or just [y,...]
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
                ys = [float(p[1]) for p in points]
            else:
                ys = [float(p) for p in points]
        except Exception:
            return

        # adapt length if needed
        try:
            if len(ys) == self.points_count:
                ys_use = ys
            elif self.has_endpoints and len(ys) == self.points_count - 2:
                ys_use = [0.0] + ys + [1.0]
            else:
                # fallback: trim/pad
                ys_use = list(ys[: self.points_count])
                if len(ys_use) < self.points_count:
                    last = ys_use[-1] if ys_use else 1.0
                    ys_use += [last] * (self.points_count - len(ys_use))
        except Exception:
            ys_use = ys

        # set endpoints toggles first (so hidden endpoint sliders are consistent)
        try:
            if self.has_endpoints:
                try:
                    self.left_toggle.setChecked(bool(round(float(ys_use[0])) == 1))
                except Exception:
                    pass
                try:
                    self.right_toggle.setChecked(bool(round(float(ys_use[-1])) == 1))
                except Exception:
                    pass
        except Exception:
            pass

        # set sliders
        try:
            for i in range(min(len(self.sliders), len(ys_use))):
                v = max(0.0, min(1.0, float(ys_use[i])))
                try:
                    self.sliders[i].blockSignals(True)
                    self.sliders[i].setValue(int(round(v * 100)))
                finally:
                    self.sliders[i].blockSignals(False)
                # keep the numeric labels in sync (signals were blocked)
                try:
                    self.value_labels[i].setText(f"{v:.2f}")
                except Exception:
                    pass
            self.update_plot()
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

        # set confidence if provided
        try:
            if confidence is not None and hasattr(self, '_vf_conf_group') and self._vf_conf_group is not None:
                cid = int(confidence)
                btn = self._vf_conf_group.button(cid)
                if btn is not None:
                    btn.setChecked(True)
        except Exception:
            pass

    def on_slider_changed(self, idx):
        val = self.sliders[idx].value() / 100.0
        self.value_labels[idx].setText(f"{val:.2f}")
        self.update_plot()

    def cycle_shape(self):
        # advance shape index and apply corresponding slider values
        self.shape_index = (self.shape_index + 1) % len(self.shape_states)
        mode = self.shape_states[self.shape_index]
        self.shape_btn.setText(mode)
        n = self.points_count
        if n == 1:
            self.sliders[0].setValue(100)
            self.update_plot()
            return
        # If endpoints exist, they are controlled by toggles; compute interior count
        interior_count = n
        if self.has_endpoints and n >= 2:
            interior_count = n - 2
        # helper to set interior values between left and right endpoints
        def set_interior_values(f_left, f_right):
            # Compute interior slider values by interpolating over the actual X positions
            # This ensures correct behaviour whether Xs are ascending or descending.
            try:
                n = self.points_count
                xs = [float(x) for x in self.xs]
                if self.has_endpoints and n >= 2 and len(xs) >= n:
                    x0 = xs[0]
                    xN = xs[-1]
                    for idx in range(1, n - 1):
                        x = xs[idx]
                        t = 0.0 if xN == x0 else (x - x0) / (xN - x0)
                        val = f_left + t * (f_right - f_left)
                        self.sliders[idx].setValue(int(max(0.0, min(1.0, val)) * 100))
                else:
                    # fallback: distribute linearly across interior_count positions
                    if interior_count <= 0:
                        return
                    if interior_count == 1:
                        v = int((f_left + f_right) / 2.0 * 100)
                        self.sliders[1 if self.has_endpoints else 0].setValue(v)
                        return
                    for j in range(interior_count):
                        frac = j / (interior_count - 1)
                        val = f_left + frac * (f_right - f_left)
                        idx = j + (1 if self.has_endpoints else 0)
                        self.sliders[idx].setValue(int(max(0.0, min(1.0, val)) * 100))
            except Exception:
                # on any error, fall back to uniform spacing
                try:
                    if interior_count <= 0:
                        return
                    if interior_count == 1:
                        v = int((f_left + f_right) / 2.0 * 100)
                        self.sliders[1 if self.has_endpoints else 0].setValue(v)
                        return
                    for j in range(interior_count):
                        frac = j / (interior_count - 1)
                        val = f_left + frac * (f_right - f_left)
                        idx = j + (1 if self.has_endpoints else 0)
                        self.sliders[idx].setValue(int(max(0.0, min(1.0, val)) * 100))
                except Exception:
                    pass

        # determine left/right endpoint values
        # determine left/right endpoint values
        left_val = 0.0
        right_val = 1.0
        if self.has_endpoints:
            try:
                # Determine desired endpoint states for this shape explicitly
                if mode == 'Linear Increasing':
                    desired_left = 0.0
                    desired_right = 1.0
                elif mode == 'Linear Decreasing':
                    desired_left = 1.0
                    desired_right = 0.0
                else:
                    # for triangular or other shapes, preserve current toggle states
                    desired_left = 1.0 if self.left_toggle.isChecked() else 0.0
                    desired_right = 1.0 if self.right_toggle.isChecked() else 0.0

                # set toggles to reflect desired endpoint states (this will also emit toggled signals)
                try:
                    self.left_toggle.setChecked(bool(round(desired_left) == 1))
                except Exception:
                    pass
                try:
                    self.right_toggle.setChecked(bool(round(desired_right) == 1))
                except Exception:
                    pass

                # explicitly set endpoint slider values and labels so they move immediately
                left_val = desired_left
                right_val = desired_right
                try:
                    self.sliders[0].setValue(int(left_val * 100))
                    self.value_labels[0].setText(f"{left_val:.2f}")
                except Exception:
                    pass
                try:
                    self.sliders[-1].setValue(int(right_val * 100))
                    self.value_labels[-1].setText(f"{right_val:.2f}")
                except Exception:
                    pass
            except Exception:
                pass

        if mode == 'Linear Increasing':
            set_interior_values(left_val, right_val)
        elif mode == 'Linear Decreasing':
            # use same left/right ordering but values will be 1->0 when left_val>right_val
            set_interior_values(left_val, right_val)
        self.update_plot()

    def _apply_endpoint_toggle(self, idx, checked):
        # idx expected 0 or points_count-1
        try:
            val = 1.0 if checked else 0.0
            self.sliders[idx].setValue(int(val * 100))
            # keep label in sync
            self.value_labels[idx].setText(f"{val:.2f}")
            self.update_plot()
        except Exception:
            pass

    def update_plot(self):
        ys = [s.value() / 100.0 for s in self.sliders]
        for i, lbl in enumerate(self.value_labels):
            try:
                lbl.setText(f"{ys[i]:.2f}")
            except Exception:
                pass
        # Sort points by X so the plotted line connects neighbors in X-order
        try:
            pairs = list(zip(self.xs, ys))
            pairs_sorted = sorted(pairs, key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = zip(*pairs_sorted)
            self._plot_line.set_xdata(xs_sorted)
            self._plot_line.set_ydata(ys_sorted)
            # update x-limits to encompass sorted Xs
            try:
                self.ax.set_xlim(min(xs_sorted), max(xs_sorted))
            except Exception:
                pass
            # update x-tick labels to show alternative names (replace commas with newline)
            try:
                if hasattr(self, 'point_labels') and self.point_labels:
                    # build labels in the same order as xs_sorted, handling possible duplicate Xs
                    labels_sorted = []
                    used = set()
                    for xval in xs_sorted:
                        found = None
                        for idx, xv in enumerate(self.xs):
                            if idx in used:
                                continue
                            try:
                                if math.isclose(float(xv), float(xval), rel_tol=1e-9, abs_tol=1e-12):
                                    found = self.point_labels[idx]
                                    used.add(idx)
                                    break
                            except Exception:
                                continue
                        labels_sorted.append(found if found is not None else '')
                    proc_labels = [(str(l).replace(',', '\n') if l is not None else '') for l in labels_sorted]
                    self.ax.set_xticks(xs_sorted)
                    self.ax.set_xticklabels(proc_labels, rotation=0, ha='center')
            except Exception:
                pass
        except Exception:
            # fallback: preserve original order
            try:
                self._plot_line.set_ydata(ys)
            except Exception:
                pass
        # adjust y-limits if necessary
        try:
            self.ax.relim()
            self.ax.autoscale_view()
        except Exception:
            pass
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def get_points(self):
        ys = [s.value() / 100.0 for s in self.sliders]
        return list(zip(self.xs, ys))

    def save(self):
        pts = self.get_points()
        if callable(self.on_save):
            try:
                conf = self._vf_conf_group.checkedId() if self._vf_conf_group is not None else None
            except Exception:
                conf = None
            self.on_save(pts, conf)

    def cancel(self):
        if callable(self.on_cancel):
            self.on_cancel()


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ranking + Value Function Elicitation (small)')
        layout = QVBoxLayout()
        # default main window size
        self.resize(_scale_px(900), _scale_px(600))
        self.setMinimumSize(_scale_px(720), _scale_px(480))

        # top area: show current criterion name (large) only
        top = QVBoxLayout()
        self.title_label = QLabel('')
        title_font = self.title_label.font()
        title_font.setPointSize(max(title_font.pointSize() + 4, 12))
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top.addWidget(self.title_label)
        layout.addLayout(top)

        # chooser button will be used at startup and then disabled to prevent changing folder
        self.load_btn = QPushButton('Choose folder and Load')
        self.load_btn.clicked.connect(self.choose_and_load_folder)
        layout.addWidget(self.load_btn)

        # central container where RankingWindow and ValueFunctionWidget will be embedded
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)
        # make container expand to take most of the window
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container_layout.setContentsMargins(_scale_px(20), _scale_px(12), _scale_px(20), _scale_px(12))
        layout.addWidget(self.container, 1)
        # bottom bar where per-view controls (confidence, action buttons) are shown
        self.bottom_bar = QWidget()
        self.bottom_bar_layout = QHBoxLayout()
        self.bottom_bar_layout.setContentsMargins(_scale_px(10), _scale_px(6), _scale_px(10), _scale_px(6))
        self.bottom_bar.setLayout(self.bottom_bar_layout)
        layout.addWidget(self.bottom_bar)

        # persistent navigation controls for qualitative multi-entry runs
        self._nav_prev_btn = QPushButton('Back')
        self._nav_next_btn = QPushButton('Next')
        self._nav_prev_btn.clicked.connect(self.nav_back)
        self._nav_next_btn.clicked.connect(self.nav_forward)

        # status label at bottom to show messages (avoid popups)
        self.status_label = QLabel('')
        # one-line status; elide if too long but provide tooltip on hover
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setWordWrap(False)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # defaults (may be overridden by folder selection)
        self.criteria_path = Path(__file__).parent / 'criteria.csv'
        self.alts_path = Path(__file__).parent / 'qualitative' / 'alternatives.csv'
        self.criteria = []
        self.alt_names = []
        self.start_points = []
        # currently visible widget in the container
        self.current_view = None
        # bookkeeping for qualitative multi-entry elicitation
        self._qual_entries = None
        self._qual_index = 0
        # track the progressive value_functions file for the current elicitation session
        self._vf_session_outpath = None
        # map base indicator -> first elicitation label (used to seed subsequent entries)
        self._first_elicitation_label_per_base = {}

        # Prompt once at startup for the working folder. If user cancels, close the app.
        # This ensures the chooser is only visible at the start.
        self.choose_and_load_folder(initial=True)
    
    def set_status(self, text: str):
        try:
            txt = str(text)
            # set full text as tooltip so hover reveals it
            try:
                self.status_label.setToolTip(txt)
                # elide text to fit current status_label width
                fm = self.status_label.fontMetrics()
                w = self.status_label.width() if self.status_label.width() > 50 else max(self.width() - 40, 200)
                elided = fm.elidedText(txt, Qt.ElideMiddle, int(w))
                self.status_label.setText(elided)
            except Exception:
                # fallback: set raw text
                self.status_label.setText(txt)
        except Exception:
            pass

    def parse_guideline_for_folder(self):
        """
        Look for a guideline.txt in the qualitative folder and parse lines of the form:
          INDICATOR: method[, [SUF1,SUF2,...]]
        Returns a dict mapping indicator base name -> {'mode': 'ranking'|'spectrum', 'suffixes': [..]}
        """
        out = {}
        # candidate locations relative to alternatives file and script
        candidates = [
            self.alts_path.parent / 'guideline.txt',
            self.alts_path.parent / 'qualitative' / 'guideline.txt',
            Path(__file__).parent / 'qualitative' / 'guideline.txt'
        ]
        gd = None
        for p in candidates:
            try:
                if p.exists():
                    gd = p
                    break
            except Exception:
                continue
        if gd is None:
            return out
        try:
            with gd.open(newline='', encoding='utf-8') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ':' not in ln:
                        continue
                    key, rest = ln.split(':', 1)
                    key = key.strip()
                    rest = rest.strip()
                    mode = 'ranking'
                    suffixes = None
                    # if rest contains a comma-separated list or a bracketed list
                    if ',' in rest:
                        parts = [p.strip() for p in rest.split(',', 1)]
                        mode_token = parts[0].lower()
                        if 'sort' in mode_token or 'spectrum' in mode_token:
                            mode = 'spectrum'
                        else:
                            mode = 'ranking'
                        # parse suffixes in the remaining part (may be bracketed)
                        if len(parts) > 1:
                            suf_part = parts[1].strip()
                            # remove surrounding brackets if present
                            if suf_part.startswith('[') and suf_part.endswith(']'):
                                suf_part = suf_part[1:-1]
                            # split by commas
                            suffixes = [s.strip() for s in suf_part.split(',') if s.strip()]
                    else:
                        mode_token = rest.lower()
                        if 'sort' in mode_token or 'spectrum' in mode_token:
                            mode = 'spectrum'
                        else:
                            mode = 'ranking'
                    out[key] = {'mode': mode, 'suffixes': suffixes}
        except Exception:
            pass
        return out

    def get_start_points_for_label(self, alts_path: Path, label: str, alt_names):
        """
        Read `alts_path` and find the row whose first cell equals `label` (exact match).
        Return a list of floats aligned to `alt_names` order when possible.
        """
        if not alts_path.exists():
            return None
        with alts_path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if r]
        if not rows:
            return None
        header = rows[0]
        # find the most recent row with matching first cell (search from bottom so later appends override earlier ones)
        target = None
        for r in reversed(rows[1:]):
            if not r:
                continue
            first = r[0].strip()
            if first == label:
                target = r
                break
        if target is None:
            # try case-insensitive match from bottom
            for r in reversed(rows[1:]):
                if not r:
                    continue
                first = r[0].strip()
                if first.lower() == (label or '').strip().lower():
                    target = r
                    break
        if target is None:
            return None
        # if header has columns after first, try to map by column name to alt_names
        vals = []
        if len(header) > 1:
            # header columns may or may not match alt_names order; prefer mapping by name if possible
            header_names = [h.strip() for h in header[1:]]
            # case-insensitive comparison
            hn_lower = [h.lower() for h in header_names]
            an_lower = [a.strip().lower() for a in alt_names]
            # if header names (excluding first) match alt_names (in some order), map by name
            if set(hn_lower) >= set([a for a in an_lower if a]):
                # build index mapping from header col -> index
                hdr_map = {h.strip().lower(): i for i, h in enumerate(header)}
                for name in alt_names:
                    idx = hdr_map.get(name.strip().lower())
                    if idx is not None and idx < len(target):
                        v = target[idx].strip()
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(v)
                    else:
                        vals.append('')
            else:
                # fallback: assume values are stored positionally after the first cell
                for i in range(1, min(len(target), 1 + len(alt_names))):
                    v = target[i].strip()
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(v)
                while len(vals) < len(alt_names):
                    vals.append('')
        else:
            # header does not have alt-name columns; fallback to taking cells 1.. as values
            for i in range(1, min(len(target), 1 + len(alt_names))):
                v = target[i].strip()
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(v)
            # pad if needed
            while len(vals) < len(alt_names):
                vals.append('')
        return vals

    def get_value_function_points_for_label(self, vf_path: Path, label: str):
        """
        Read `value_functions.csv` and return elicited points for the row whose name matches `label`.
        Returns list of (x,y) pairs or None.
        """
        if not vf_path.exists():
            return None
        # Read all rows and prefer the last matching entry if duplicates exist
        with vf_path.open(newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            last_match = None
            for r in reader:
                name = (r.get('name') or '').strip()
                if not name:
                    continue
                if name == label or name.lower() == label.lower():
                    last_match = r
            if last_match is None:
                return None
            pts_raw = last_match.get('elicited_points')
            if not pts_raw:
                return None
            try:
                pts = json.loads(pts_raw)
                ptsf = [(float(x), float(y)) for x, y in pts]
                return ptsf
            except Exception:
                return None
        # not found
        return None

    def get_value_function_confidence_for_label(self, vf_path: Path, label: str):
        """
        Read `value_functions.csv` and return the `confidence` field (if present)
        for the last matching entry whose name matches `label`. Returns int or None.
        """
        if not vf_path.exists():
            return None
        with vf_path.open(newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            last_match = None
            for r in reader:
                name = (r.get('name') or '').strip()
                if not name:
                    continue
                if name == label or name.lower() == label.lower():
                    last_match = r
            if last_match is None:
                return None
            conf_raw = last_match.get('confidence')
            if conf_raw is None or conf_raw == '':
                return None
            try:
                return int(float(conf_raw))
            except Exception:
                try:
                    return int(conf_raw)
                except Exception:
                    return None

    def clear_container(self):
        # remove all widgets from the container layout
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass
        self.current_view = None

    def show_view(self, widget: QWidget):
        # Hide previous view and show the new widget inside the container.
        try:
            if self.current_view is not None and self.current_view is not widget:
                try:
                    self.current_view.hide()
                except Exception:
                    pass
            # If widget is not already in the container, add it
            if widget.parent() is not self.container:
                widget.setParent(self.container)
                # add with stretch so the widget takes available space
                self.container_layout.addWidget(widget, 1)
            else:
                # ensure widget occupies stretch index if already present
                try:
                    idx = self.container_layout.indexOf(widget)
                    if idx >= 0:
                        self.container_layout.setStretch(idx, 1)
                except Exception:
                    pass
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.show()
            self.current_view = widget
            # set the main title and status depending on widget type
            try:
                # RankingWindow: show indicator name as title and instructions in status
                if hasattr(widget, 'indicator_name') and widget.indicator_name:
                    self.title_label.setText(str(widget.indicator_name))
                    instr = f"Indicator: {widget.indicator_name} — Reorder alternatives by dragging. Select multiple and click 'Tie Selected' to place them at the same rank."
                    self.set_status(instr)
                # ValueFunctionWidget: show criterion name as title
                elif hasattr(widget, 'criterion') and isinstance(widget.criterion, dict):
                    self.title_label.setText(str(widget.criterion.get('name', '')))
                    self.set_status('Adjust the value function using the sliders. Click Next to save and continue.')
                else:
                    # default: clear title
                    self.title_label.setText('')
            except Exception:
                pass
            # also set the OS window title so it is visible in the titlebar
            try:
                base = 'Ranking + Value Function Elicitation (small)'
                extra = self.title_label.text().strip()
                if extra:
                    full = f"{base} - {extra}"
                else:
                    full = base
                # if the widget exposes a mode (e.g., RankingWindow.mode), append it
                if hasattr(widget, 'mode') and getattr(widget, 'mode'):
                    full = f"{full} [{getattr(widget, 'mode')}]"
                self.setWindowTitle(full)
            except Exception:
                pass
            # manage main bottom bar: clear previous per-view controls and show new ones
            try:
                # remove existing widgets from bottom bar
                while self.bottom_bar_layout.count():
                    it = self.bottom_bar_layout.takeAt(0)
                    w = it.widget()
                    if w is not None:
                        try:
                            w.setParent(None)
                        except Exception:
                            pass
                # if the widget exposes a bottom_panel, reparent and add it to the bottom bar
                if hasattr(widget, 'bottom_panel') and getattr(widget, 'bottom_panel') is not None:
                    try:
                        bp = widget.bottom_panel
                        bp.setParent(self.bottom_bar)
                        self.bottom_bar_layout.addWidget(bp)
                    except Exception:
                        pass

                # add qualitative navigation buttons (Back/Next) on the right
                if self._qual_entries is not None:
                    try:
                        self.bottom_bar_layout.addStretch(1)
                    except Exception:
                        pass
                    try:
                        self._nav_prev_btn.setParent(self.bottom_bar)
                        self._nav_next_btn.setParent(self.bottom_bar)
                        self.bottom_bar_layout.addWidget(self._nav_prev_btn)
                        self.bottom_bar_layout.addWidget(self._nav_next_btn)
                    except Exception:
                        pass
                    try:
                        self._update_nav_buttons_state()
                    except Exception:
                        pass
            except Exception:
                pass

            # force layout and repaint
            self.container.updateGeometry()
            self.container.repaint()
            self.update()
            QApplication.processEvents()
        except Exception:
            pass

    def _get_vf_paths_for_lookup(self):
        paths = []
        try:
            session_path = getattr(self, '_vf_session_outpath', None)
            if session_path:
                paths.append(Path(session_path))
        except Exception:
            pass
        try:
            paths.append(Path(self.criteria_path).parent / 'value_functions.csv')
        except Exception:
            pass
        # de-duplicate
        out = []
        seen = set()
        for p in paths:
            try:
                ps = str(p)
            except Exception:
                continue
            if ps in seen:
                continue
            seen.add(ps)
            out.append(p)
        return out

    def _refresh_entry_from_files(self, ent: dict):
        """Refresh start_points and vf_points for an entry from saved files if possible."""
        if not ent:
            return
        label = str(ent.get('indicator') or '').strip()
        if not label:
            return
        # ranking/scoring
        try:
            sp = self.get_start_points_for_label(self.alts_path, label, self.alt_names)
            if sp is not None:
                ent['start_points'] = sp
        except Exception:
            pass
        # value function points + confidence
        for pth in self._get_vf_paths_for_lookup():
            try:
                if not pth or not pth.exists():
                    continue
                pts = self.get_value_function_points_for_label(pth, label)
                if pts is not None:
                    ent['vf_points'] = pts
                conf = self.get_value_function_confidence_for_label(pth, label)
                if conf is not None:
                    ent['vf_confidence'] = conf
                if pts is not None or conf is not None:
                    break
            except Exception:
                continue

    def _update_nav_buttons_state(self):
        try:
            if self._qual_entries is None:
                self._nav_prev_btn.setVisible(False)
                self._nav_next_btn.setVisible(False)
                return
            self._nav_prev_btn.setVisible(True)
            self._nav_next_btn.setVisible(True)
            # Back is enabled if we can go to previous indicator or if we are on VF and can go back to ranking
            can_prev = bool(self._qual_index > 0)
            # if currently in VF editor, allow Back even at index 0 (back to ranking)
            if not can_prev and isinstance(self.current_view, ValueFunctionWidget):
                can_prev = True
            self._nav_prev_btn.setEnabled(can_prev)
            # Next is always available during a qualitative run:
            # - on ranking: Next saves scoring + opens VF
            # - on VF: Next saves VF + advances (or finishes)
            self._nav_next_btn.setEnabled(True)
        except Exception:
            pass

    def _show_indicator_index(self, idx: int):
        if self._qual_entries is None:
            return
        if idx < 0 or idx >= len(self._qual_entries):
            return
        self._qual_index = idx
        ent = self._qual_entries[self._qual_index]
        try:
            self._refresh_entry_from_files(ent)
        except Exception:
            pass
        sp = ent.get('start_points')
        # if None, try reading from file now
        if sp is None:
            try:
                sp = self.get_start_points_for_label(self.alts_path, ent.get('indicator'), self.alt_names)
            except Exception:
                sp = None
        rw = RankingWindow(self.alt_names, sp, indicator_name=ent.get('indicator'), mode=ent.get('mode', 'ranking'))
        rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.show_view(rw)

    def nav_back(self):
        """Back: if currently editing VF, go back to ranking for the same indicator; otherwise go to previous indicator."""
        if self._qual_entries is None:
            return
        try:
            if isinstance(self.current_view, ValueFunctionWidget):
                # back to ranking of current indicator
                self._show_indicator_index(self._qual_index)
                return
        except Exception:
            pass
        if self._qual_index > 0:
            self._show_indicator_index(self._qual_index - 1)

    def nav_forward(self):
        """Next: always saves something.

        - From ranking: saves scoring and opens the VF editor.
        - From VF: saves the VF and advances to the next indicator (or finishes).
        """
        if self._qual_entries is None:
            return
        # If we are in the VF editor, saving it also advances the index.
        try:
            if isinstance(self.current_view, ValueFunctionWidget):
                self.current_view.save()
                return
        except Exception:
            pass

        # Otherwise, we are in the ranking view: save scoring and open the VF editor.
        try:
            if isinstance(self.current_view, RankingWindow):
                ranks = self.current_view.get_ranks()
                conf = None
                try:
                    conf = self.current_view.get_confidence()
                except Exception:
                    conf = None
                indicator_name = getattr(self.current_view, 'indicator_name', None)
                self.open_value_editor(ranks, indicator_name=indicator_name, confidence=conf)
                return
        except Exception:
            pass

        # Fallback: if view is unexpected, advance without saving.
        if self._qual_index < len(self._qual_entries) - 1:
            self._show_indicator_index(self._qual_index + 1)

    def choose_and_load_folder(self, initial: bool = False):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder containing criteria.csv and alternatives.csv', str(Path(__file__).parent))
        if not folder:
            # if this was the initial prompt, close the app; otherwise just return
            if initial:
                # avoid a popup; just close
                self.close()
            return
        folderp = Path(folder)
        self.criteria_path = folderp / 'criteria.csv'
        self.alts_path = folderp / 'alternatives.csv'
        # update top labels so user sees chosen folder
        try:
            # Move folder info to the status label at the bottom
            status_text = f"Criteria: {self.criteria_path} | Alternatives: {self.alts_path}"
            self.set_status(status_text)
        except Exception:
            pass
        self.load_files()
        # disable the chooser so folder cannot be changed during the run
        self.load_btn.setEnabled(False)
        self.load_btn.setVisible(False)

    def _split_base_suffix(self, label: str):
        try:
            s = str(label).strip()
            if ' - ' not in s:
                return None, None
            base, suf = [p.strip() for p in s.rsplit(' - ', 1)]
            if not base or not suf or not re.match(r'^[A-Za-z]{2,3}$', suf):
                return None, None
            return base, suf.upper()
        except Exception:
            return None, None

    def _get_copy_candidates(self, current_label: str, kind: str):
        """Return a list of other labels with same base that have data to copy."""
        base, _ = self._split_base_suffix(current_label)
        if not base:
            return []

        # build candidate universe from known entries
        universe = []
        try:
            if self._qual_entries is not None:
                universe = [str(e.get('indicator', '')).strip() for e in self._qual_entries if e.get('indicator')]
        except Exception:
            universe = []

        # also allow copying from the base name itself (if present in saved files)
        universe.append(base)

        # de-duplicate while keeping order
        seen = set()
        uniq = []
        for u in universe:
            if not u or u == current_label:
                continue
            if u in seen:
                continue
            # only allow siblings of the same base indicator
            ub, _us = self._split_base_suffix(u)
            if ub is not None:
                if ub.lower() != base.lower():
                    continue
            else:
                # non-suffixed labels must be exactly the base (avoid other indicators)
                if str(u).strip().lower() != base.lower():
                    continue
            seen.add(u)
            uniq.append(u)

        # filter by availability of data
        out = []
        vf_paths = []
        try:
            # prefer the current session file (value_functions_*.csv) if set
            session_path = getattr(self, '_vf_session_outpath', None)
            if session_path:
                vf_paths.append(Path(session_path))
        except Exception:
            pass
        vf_paths.append(Path(self.criteria_path).parent / 'value_functions.csv')
        for u in uniq:
            try:
                if kind == 'ranking':
                    sp = None
                    try:
                        sp = self.get_start_points_for_label(self.alts_path, u, self.alt_names)
                    except Exception:
                        sp = None
                    # fallback to in-memory start_points if present
                    if sp is None and self._qual_entries is not None:
                        try:
                            ent = next((e for e in self._qual_entries if str(e.get('indicator')) == u), None)
                            if ent is not None and ent.get('start_points') is not None:
                                sp = ent.get('start_points')
                        except Exception:
                            sp = None
                    if sp is not None:
                        out.append(u)
                elif kind == 'value_function':
                    pts = None
                    try:
                        if self._qual_entries is not None:
                            ent = next((e for e in self._qual_entries if str(e.get('indicator')) == u), None)
                            if ent is not None and ent.get('vf_points'):
                                pts = ent.get('vf_points')
                    except Exception:
                        pts = None
                    if pts is None:
                        for pth in vf_paths:
                            try:
                                if pth and pth.exists():
                                    pts = self.get_value_function_points_for_label(pth, u)
                                    if pts is not None:
                                        break
                            except Exception:
                                pts = None
                    if pts is not None:
                        out.append(u)
            except Exception:
                continue
        return out

    def copy_from_for_current_indicator(self, kind: str = 'ranking'):
        """Invoked by embedded views. kind is 'ranking' or 'value_function'."""
        try:
            # determine current label from the visible view
            current_label = None
            if self.current_view is None:
                return
            if kind == 'ranking' and hasattr(self.current_view, 'indicator_name'):
                current_label = str(getattr(self.current_view, 'indicator_name') or '').strip()
            elif kind == 'value_function' and hasattr(self.current_view, 'criterion'):
                try:
                    current_label = str(self.current_view.criterion.get('name', '')).strip()
                except Exception:
                    current_label = None

            if not current_label:
                return

            base, _ = self._split_base_suffix(current_label)
            if not base:
                self.set_status('Copy from is available only for country-suffixed indicators ("Base - CC").')
                return

            candidates = self._get_copy_candidates(current_label, kind)
            if not candidates:
                self.set_status('No eligible source found to copy from (save another country first).')
                return

            src, ok = QInputDialog.getItem(self, 'Copy from', 'Select source indicator:', candidates, 0, False)
            if not ok or not src:
                return
            src = str(src).strip()

            if kind == 'ranking':
                sp = self.get_start_points_for_label(self.alts_path, src, self.alt_names)
                if sp is None:
                    # fallback to in-memory
                    if self._qual_entries is not None:
                        ent = next((e for e in self._qual_entries if str(e.get('indicator')) == src), None)
                        sp = ent.get('start_points') if ent is not None else None
                if sp is None:
                    self.set_status(f'No ranking/scoring found for {src}.')
                    return
                try:
                    if hasattr(self.current_view, 'apply_start_points'):
                        self.current_view.apply_start_points(sp)
                except Exception:
                    pass
                # keep in-memory entry aligned
                try:
                    if self._qual_entries is not None:
                        ent_cur = next((e for e in self._qual_entries if str(e.get('indicator')) == current_label), None)
                        if ent_cur is not None:
                            ent_cur['start_points'] = sp
                except Exception:
                    pass
                self.set_status(f'Copied ranking from {src} → {current_label}.')
                return

            if kind == 'value_function':
                # prefer the current session output file, then the canonical value_functions.csv
                vf_paths = []
                try:
                    session_path = getattr(self, '_vf_session_outpath', None)
                    if session_path:
                        vf_paths.append(Path(session_path))
                except Exception:
                    pass
                vf_paths.append(Path(self.criteria_path).parent / 'value_functions.csv')
                pts = None
                conf = None
                try:
                    if self._qual_entries is not None:
                        ent = next((e for e in self._qual_entries if str(e.get('indicator')) == src), None)
                        if ent is not None and ent.get('vf_points'):
                            pts = ent.get('vf_points')
                        if ent is not None and ent.get('vf_confidence') is not None:
                            conf = ent.get('vf_confidence')
                except Exception:
                    pts = None
                if pts is None:
                    for pth in vf_paths:
                        try:
                            if pth and pth.exists():
                                pts = self.get_value_function_points_for_label(pth, src)
                                if pts is not None:
                                    break
                        except Exception:
                            pts = None
                if conf is None:
                    for pth in vf_paths:
                        try:
                            if pth and pth.exists():
                                conf = self.get_value_function_confidence_for_label(pth, src)
                                if conf is not None:
                                    break
                        except Exception:
                            conf = None
                if pts is None:
                    self.set_status(f'No value function found for {src}.')
                    return
                try:
                    if hasattr(self.current_view, 'apply_points'):
                        self.current_view.apply_points(pts, confidence=conf)
                except Exception:
                    pass
                # keep in-memory entry aligned
                try:
                    if self._qual_entries is not None:
                        ent_cur = next((e for e in self._qual_entries if str(e.get('indicator')) == current_label), None)
                        if ent_cur is not None:
                            ent_cur['vf_points'] = pts
                            if conf is not None:
                                ent_cur['vf_confidence'] = conf
                except Exception:
                    pass
                self.set_status(f'Copied value function from {src} → {current_label}.')
                return
        except Exception:
            pass

    def load_files(self):
        self.criteria = read_criteria(self.criteria_path)
        alts_result = read_alternatives(self.alts_path)
        if not alts_result:
            self.set_status(f'No alternatives found in {self.alts_path}')
            return
        self.alt_names = alts_result[0]
        entries = alts_result[1]

        # Qualitative format: entries is a list of {'indicator', 'raw_label', 'start_points'}
        if isinstance(entries, list) and entries and isinstance(entries[0], dict) and 'start_points' in entries[0]:
            # parse guideline (if present) to determine default mode and multiple elicitations
            guideline = self.parse_guideline_for_folder()
            expanded = []
            for ent in entries:
                base = ent.get('indicator')
                cfg = guideline.get(base, {})
                mode = cfg.get('mode', 'ranking')
                suffixes = cfg.get('suffixes')
                if suffixes:
                    # create one entry per suffix; first uses original start_points,
                    # others should default to the same guideline start_points (copying is manual via "Copy from...")
                    for i, suf in enumerate(suffixes):
                        lbl = f"{base} - {suf}"
                        expanded.append({'indicator': lbl, 'raw_label': ent.get('raw_label'), 'start_points': ent.get('start_points'), 'mode': mode})
                else:
                    expanded.append({'indicator': base, 'raw_label': ent.get('raw_label'), 'start_points': ent.get('start_points'), 'mode': mode})

            self._qual_entries = expanded
            self._qual_index = 0
            # try to seed any existing value functions for these indicators
            try:
                vf_path = Path(self.criteria_path).parent / 'value_functions.csv'
                if vf_path.exists():
                    for ent in self._qual_entries:
                        try:
                            name = ent.get('indicator') or ''
                            raw = ent.get('raw_label') or ''
                            pts = None
                            # try exact indicator name first
                            if name:
                                pts = self.get_value_function_points_for_label(vf_path, name)
                            # try raw label
                            if pts is None and raw:
                                pts = self.get_value_function_points_for_label(vf_path, raw)
                            # try base name before any ' - ' suffix
                            if pts is None and name and ' - ' in name:
                                base = name.split(' - ')[0].strip()
                                pts = self.get_value_function_points_for_label(vf_path, base)
                            # as a last resort try the first token
                            if pts is None and name:
                                base = name.split('-')[0].strip()
                                pts = self.get_value_function_points_for_label(vf_path, base)
                            if pts is not None:
                                ent['vf_points'] = pts
                                try:
                                    conf = self.get_value_function_confidence_for_label(vf_path, name)
                                    if conf is not None:
                                        ent['vf_confidence'] = conf
                                except Exception:
                                    pass
                        except Exception:
                            continue
            except Exception:
                pass
            # start with first indicator entry
            ent0 = self._qual_entries[self._qual_index]
            detected_mode = ent0.get('mode', 'ranking')
            # ensure we have start_points for the first entry
            sp = ent0.get('start_points')
            rw = RankingWindow(self.alt_names, sp, indicator_name=ent0.get('indicator'), mode=detected_mode)
            rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.show_view(rw)
        else:
            # non-qualitative: entries is a simple list of start points
            self.start_points = entries
            rw = RankingWindow(self.alt_names, self.start_points)
            rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.show_view(rw)

    def open_value_editor(self, ranks, indicator_name=None, confidence=None):
        # compute integer scoring values based on distinct ranks (places)
        # Scores range from 1 .. P where P = number of distinct ranks
        # e.g. 4 alternatives -> ranks (no ties) -> scores 1,2,3,4
        num_ranks = len(ranks)
        if num_ranks <= 0:
            self.set_status('No ranks found; cannot compute scoring')
            return
        # Determine criterion bounds so scoring and VF x-positions respect criteria
        crit = None
        if indicator_name:
            for c in self.criteria:
                if str(c.get('name')).strip().lower() == str(indicator_name).strip().lower():
                    crit = c
                    break
        if crit is None:
            crit = {'name': indicator_name or 'Indicator', 'min': 0.0, 'max': 1.0, 'group': ''}

        alt_score = {}
        # assign scores so that topmost rank (idx=0) gets the criterion max,
        # and bottommost gets the criterion min; equally spaced between them
        # Ensure criterion bounds are ordered so that top rank maps to the larger value.
        try:
            raw_min = float(crit.get('min', 0.0))
        except Exception:
            raw_min = 0.0
        try:
            raw_max = float(crit.get('max', 1.0))
        except Exception:
            raw_max = 1.0
        cmin = min(raw_min, raw_max)
        cmax = max(raw_min, raw_max)
        if num_ranks == 1:
            score_for_rank = [(cmin + cmax) / 2.0]
        else:
            step = (cmax - cmin) / (num_ranks - 1)
            # rank 0 -> cmax, rank 1 -> cmax - step, ..., rank N-1 -> cmin
            score_for_rank = [cmax - i * step for i in range(num_ranks)]
        for idx, rank in enumerate(ranks):
            val = score_for_rank[idx]
            for a in rank:
                alt_score[a] = val

        # write scoring into alternatives.csv as a new row
        # determine label for this elicitation. If we have an indicator name, use QI - E# pattern
        label = 'Computed scoring'
        if indicator_name:
            # Use the indicator name itself as the scoring label. Do not create '-E#' variants.
            label = indicator_name
        try:
            # remember the label we will write so subsequent elicitations can use it as seed
            self._last_elicitation_label = label
            # pass along indicator-level confidence (may be None)
            # If the label already exists in the alternatives file, append_scoring_to_alternatives
            # will overwrite the existing row rather than adding a new '-E#' row.
            self.append_scoring_to_alternatives(self.alts_path, self.alt_names, alt_score, label=label, confidence=confidence)
        except Exception as e:
            # show error in status label instead of popup
            self.set_status(f'Could not append scoring: {e}')
            return

        # open value function dialog per criterion
        vf_rows = []
        # try to match a criterion by indicator_name; if not found, present a generic dialog
        crit = None
        if indicator_name:
            for c in self.criteria:
                if str(c.get('name')).strip().lower() == str(indicator_name).strip().lower():
                    crit = c
                    break
        if crit is None:
            # create a minimal criterion from indicator name
            crit = {'name': indicator_name or 'Indicator', 'min': 0.0, 'max': 1.0, 'group': ''}

        # embed the ValueFunctionWidget inside the main window's container
        pts_store = {}

        def _on_save(pts, vf_confidence=None):
            # write value functions and update status label
            vf_name = indicator_name or crit.get('name')
            # determine base indicator (e.g., 'QI2' from 'QI2 - IT') and inherit its group
            base = (indicator_name or '').split('-')[0].strip()
            base_group = ''
            try:
                if base:
                    bc = next((c for c in self.criteria if str(c.get('name')).strip().lower() == base.lower()), None)
                    if bc is not None:
                        base_group = bc.get('group', '')
            except Exception:
                base_group = ''
            # include VF-level confidence if provided
            vf_rows_local = [(vf_name, pts, base_group, vf_confidence)]
            out_path = Path(self.criteria_path).parent / 'value_functions.csv'
            try:
                # if we are already in a session, reuse the same progressive file
                if getattr(self, '_vf_session_outpath', None) is None:
                    written = write_value_functions(out_path, vf_rows_local)
                    # store session path for subsequent saves during this elicitation run
                    try:
                        self._vf_session_outpath = Path(written)
                    except Exception:
                        self._vf_session_outpath = Path(written)
                else:
                    # write into the already-chosen session file
                    written = write_value_functions(self._vf_session_outpath, vf_rows_local)
            except Exception as e:
                self.set_status(f'Failed to write value functions: {e}')
                return
            # write_value_functions returns the actual file path written
            try:
                display_path = written if isinstance(written, str) else str(written)
            except Exception:
                display_path = str(out_path)
            self.set_status(f'Value functions saved to {display_path}; scoring appended to {self.alts_path}')
            # proceed to next qualitative entry if present
            if self._qual_entries is not None:
                self._qual_index += 1
                if self._qual_index < len(self._qual_entries):
                    ent = self._qual_entries[self._qual_index]
                    sp = ent.get('start_points')
                    # replace container contents with next ranking, respecting the mode from guideline
                    rw = RankingWindow(self.alt_names, sp, indicator_name=ent.get('indicator'), mode=ent.get('mode', 'ranking'))
                    rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    self.show_view(rw)
                else:
                    # finished qualitative run — prompt only for closing
                    resp = QMessageBox.question(self, 'All done', 'Completed elicitation for all indicators in the folder. Close application?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if resp == QMessageBox.Yes:
                        self.close()
                    else:
                        self.set_status('Elicitation complete. You may close the application when ready.')
                        self._qual_entries = None
                        self._qual_index = 0
            else:
                # non-qualitative flow finished
                self.set_status('Elicitation complete.')

        def _on_cancel():
            self.set_status('Value function elicitation cancelled; partial results not saved.')
            # do not show popups; leave container empty
            try:
                self._vf_session_outpath = None
            except Exception:
                pass

        # show the embedded editor (replacing the ranking view)
        # For qualitative indicators, include fixed endpoints at criterion min/max
        # Points shown = num_ranks interior points + 2 endpoints
        points_with_endpoints = num_ranks + 2
        xs_override = None
        if indicator_name is not None:
            try:
                # For qualitative elicitation, X positions must be integer ranks
                # starting at 1 and ending at the number of alternatives.
                cmin = 1.0
                cmax = float(len(self.alt_names)) if self.alt_names else float(crit.get('max', 1.0))
                if num_ranks <= 0:
                    xs_override = [cmin, cmax]
                else:
                    # spacing between points including the two endpoints
                    step_x = (cmax - cmin) / (num_ranks + 1)
                    # interior points correspond to ranked items; ensure the first (top) rank
                    # maps to the highest X (near cmax). Compute interior positions from
                    # cmin+step_x .. cmin+num_ranks*step_x, then reverse to make descending.
                    interior = [cmin + i * step_x for i in range(1, num_ranks + 1)]
                    interior_desc = list(reversed(interior))
                    xs_override = [cmin] + interior_desc + [cmax]
            except Exception:
                # fallback: use 1..N numbering for qualitative ranks (not 0..N-1)
                xs_override = [float(i) for i in range(1, points_with_endpoints + 1)]

        # attempt to seed the value function points: prefer in-memory pre-seeded `vf_points`
        ys_override = None
        # check current qual_entries for this exact indicator name
        if self._qual_entries is not None and indicator_name is not None:
            try:
                ent_lookup = next((e for e in self._qual_entries if e.get('indicator') == indicator_name), None)
                if ent_lookup is not None:
                    vfp = ent_lookup.get('vf_points')
                    if vfp:
                        # normalize vfp to include endpoints: vfp may contain only interior ys
                        try:
                            existing_ys = [float(y) for x, y in vfp]
                        except Exception:
                            existing_ys = [float(y) for x, y in vfp if isinstance(y, (int, float))]
                        if len(existing_ys) == num_ranks:
                            ys_override = [0.0] + existing_ys + [1.0]
                        elif len(existing_ys) == points_with_endpoints:
                            ys_override = existing_ys
                        else:
                            ys_override = [i / (points_with_endpoints - 1) for i in range(points_with_endpoints)]
                        if xs_override is None:
                            xs_override = [float(x) for x, y in vfp]
            except Exception:
                pass
        # otherwise fall back to reading last saved VF from file (seed_label)
        if ys_override is None:
            seed_label = getattr(self, '_last_elicitation_label', None)
            vf_path = Path(self.criteria_path).parent / 'value_functions.csv'
            if seed_label and vf_path.exists():
                try:
                    pts = self.get_value_function_points_for_label(vf_path, seed_label)
                    if pts:
                        try:
                            existing_ys = [float(y) for x, y in pts]
                        except Exception:
                            existing_ys = [float(y) for x, y in pts if isinstance(y, (int, float))]
                        if len(existing_ys) == num_ranks:
                            ys_override = [0.0] + existing_ys + [1.0]
                        elif len(existing_ys) == points_with_endpoints:
                            ys_override = existing_ys
                        else:
                            ys_override = [i / (points_with_endpoints - 1) for i in range(points_with_endpoints)]
                        if xs_override is None:
                            xs_override = [float(x) for x, y in pts]
                except Exception:
                    pass

        # include endpoint labels
        point_labels = [', '.join(rank) for rank in ranks]
        point_labels_with_endpoints = ['Min'] + point_labels + ['Max']
        vf_widget = ValueFunctionWidget(crit, points_with_endpoints, xs_override=xs_override, ys_override=ys_override, point_labels=point_labels_with_endpoints, on_save=_on_save, on_cancel=_on_cancel, parent=self.container)
        self.show_view(vf_widget)

    def append_scoring_to_alternatives(self, alts_path: Path, alt_names, alt_score_map, label='Computed scoring', confidence=None):
        # read everything, then append a row where first cell describes the row and following are scores in the same column order
        # This function will ensure a top-level 'confidence' column exists in the header and will write
        # the provided confidence value for the appended scoring row.
        rows = []
        with alts_path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [r for r in reader]

        # determine header-style columns
        if not rows:
            raise RuntimeError('Alternatives file empty')
        header = rows[0]
        header_cols = [h for h in header[1:]] if len(header) > 1 else []
        # ensure 'confidence' column exists in header (case-insensitive)
        hdr_lower = [h.strip().lower() for h in header]
        if 'confidence' not in hdr_lower:
            # append confidence column name and pad existing rows with empty cell
            rows[0] = rows[0] + ['confidence']
            for i in range(1, len(rows)):
                rows[i] = rows[i] + ['']
            header_cols = [h for h in rows[0][1:]]
        # Prepare a mapping for case-insensitive name matching from alt_score_map
        score_map_ci = {k.strip().lower(): v for k, v in alt_score_map.items()}

        if header_cols:
            # Build a row that aligns with the existing header columns, leaving non-matching columns blank
            new_row = [label]
            for col in header_cols:
                key = col.strip()
                if key and key.strip().lower() == 'confidence':
                    new_row.append(str(confidence) if confidence is not None else '')
                elif key and key.strip().lower() in score_map_ci:
                    new_row.append(str(score_map_ci[key.strip().lower()]))
                else:
                    new_row.append('')
            # If a row for this label already exists, overwrite it; otherwise append
            replaced = False
            for i in range(1, len(rows)):
                try:
                    first = rows[i][0].strip() if rows[i] and rows[i][0] is not None else ''
                except Exception:
                    first = ''
                if first.lower() == (label or '').strip().lower():
                    # normalize length to header length (including first cell)
                    rows[i] = new_row
                    replaced = True
                    break
            if not replaced:
                rows.append(new_row)
        else:
            # No header columns to align to; fall back to using provided alt_names order
            scores = []
            for name in alt_names:
                val = None
                if name is not None:
                    val = score_map_ci.get(name.strip().lower())
                if val is None:
                    scores.append('')
                else:
                    scores.append(str(val))
            # append confidence at end
            new_row = [label] + scores + ([str(confidence)] if confidence is not None else [''])
            rows.append(new_row)

        with alts_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def next_elicitation_label(self, alts_path: Path, indicator_base: str):
        # find existing rows that start with e.g. 'QI1 - E1' and compute next index
        existing = []
        with alts_path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                first = r[0].strip()
                if first.lower().startswith(indicator_base.strip().lower() + ' - e'):
                    existing.append(first)
        # count existing E entries
        maxn = 0
        for e in existing:
            parts = e.split('-')
            if len(parts) >= 2:
                tail = parts[-1].strip()
                if tail.upper().startswith('E'):
                    try:
                        n = int(tail[1:])
                        if n > maxn:
                            maxn = n
                    except Exception:
                        pass
        return f"{indicator_base} - E{maxn+1}"


def write_value_functions(path: Path, vf_rows):
    # vf_rows: list of (name, [(x,y),...]) or (name, [(x,y)...], group)
    # We no longer save executable/lambda expressions to the CSV. Persist only
    # elicited points and metadata so callers can rebuild safe interpolators.
    header = ['name', 'group', 'elicited_points', 'confidence', 'elicitation_meta']
    # If callers pass the base 'value_functions.csv' path we will not overwrite
    # that base file; instead, write to the next progressive numbered file
    # `value_functions_1.csv`, `value_functions_2.csv`, ... in the same folder.
    actual_path = Path(path)
    out_path = actual_path

    names_to_replace = { (item[0] if len(item) >= 1 else '').strip().lower() for item in vf_rows }
    existing_rows = []
    if out_path.exists():
        try:
            with out_path.open(newline='', encoding='utf-8') as f:
                rdr = list(csv.reader(f))
                if len(rdr) > 0:
                    existing_header = rdr[0]
                    existing_rows = rdr[1:]
                else:
                    existing_header = header
        except Exception:
            existing_rows = []

    # filter out any existing rows matching the names
    filtered = []
    for r in existing_rows:
        if not r:
            continue
        first = (r[0].strip().lower() if r and r[0] is not None else '')
        if first in names_to_replace:
            continue
        filtered.append(r)

    # now write header, filtered rows, and new rows
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in filtered:
            try:
                name = r[0] if len(r) > 0 else ''
                group = r[1] if len(r) > 1 else ''
                elicited = r[2] if len(r) > 2 else ''
                confidence = ''
                meta = ''
                if len(r) >= 5:
                    confidence = r[3]
                    meta = r[4]
                elif len(r) == 4:
                    cand = (r[3] or '').strip()
                    if cand.startswith('{') or cand.startswith('['):
                        meta = cand
                    else:
                        confidence = cand
                writer.writerow([name, group, elicited, confidence, meta])
            except Exception:
                # skip malformed row
                continue

        for item in vf_rows:
            # items may be (name, pts), (name, pts, group) or (name, pts, group, confidence)
            name = ''
            pts = []
            group = ''
            confidence_val = ''
            if len(item) == 4:
                name, pts, group, confidence_val = item
            elif len(item) == 3:
                name, pts, group = item
            elif len(item) == 2:
                name, pts = item
            else:
                try:
                    name = item[0]
                    pts = item[1]
                except Exception:
                    continue

            writer.writerow([
                name,
                group,
                json.dumps([[float(x), float(y)] for x, y in pts]),
                (str(confidence_val) if confidence_val is not None else ''),
                json.dumps({'mode': 'Manual', 'notes': 'elicited via ranking_value_ui'})
            ])
    return str(out_path)


def main():
    _maybe_force_xcb_on_wayland()
    app = QApplication(sys.argv)
    apply_ui_scale(app)
    w = MainApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
