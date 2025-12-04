# main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QWidget, QSlider, QDoubleSpinBox
)
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QDoubleValidator
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from plotter import Plotter

class MainWindow(QMainWindow):
    def __init__(self, elicitation_process):
        super().__init__()
        self.process = elicitation_process
        self.plotter = Plotter()
        self.setWindowTitle("Value Function Elicitation")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # File upload section
        self.file_label = QLabel("No file loaded")
        self.upload_button = QPushButton("Upload CSV")
        self.upload_button.clicked.connect(self.upload_csv)
        # Optional loader function remains available; button removed per user request

        # Attribute info
        self.attr_label = QLabel("No attribute selected")

    # (manual single-point UI removed — thresholds and shape controls handle point creation)

        # Monotonic / Non-monotonic selector
        self.mono_selector = QComboBox()
        self.mono_selector.addItems(["Monotonic", "Non-monotonic"])
        self.mono_selector.currentIndexChanged.connect(self.on_mono_changed)

        # Shape selector for non-monotonic: concave (∩) or convex (∪)
        self.shape_selector = QComboBox()
        self.shape_selector.addItems(["Concave (∩)", "Convex (∪)"])
        self.shape_selector.setVisible(False)
        self.shape_selector.currentIndexChanged.connect(self.on_shape_changed)

        # Threshold inputs
        self.x_increase_input = QLineEdit()
        self.x_increase_input.setPlaceholderText("X after which increasing X not important (default = max)")
        self.x_decrease_input = QLineEdit()
        self.x_decrease_input.setPlaceholderText("X after which decreasing X not important (default = min)")

        # (textChanged connections moved after indifference/peak widgets are created)

        # Indifference / peak inputs
        # For monotonic: single indifference point x0.5
        self.indiff_input = QLineEdit()
        self.indiff_input.setPlaceholderText("Indifference point x0.5 (optional)")
        # Additional monotonic indifference points: 0.25 and 0.75
        self.indiff25_input = QLineEdit()
        self.indiff25_input.setPlaceholderText("Indifference x0.25 (optional)")
        self.indiff75_input = QLineEdit()
        self.indiff75_input.setPlaceholderText("Indifference x0.75 (optional)")
        # For non-monotonic: peak/valley location and left/right indifference points
        self.peak_location_input = QLineEdit()
        self.peak_location_input.setPlaceholderText("Peak/Valley location (optional)")
        self.left_indiff_input = QLineEdit()
        self.left_indiff_input.setPlaceholderText("Left indifference x (optional)")
        self.right_indiff_input = QLineEdit()
        self.right_indiff_input.setPlaceholderText("Right indifference x (optional)")

        # connect inputs so changes automatically update the plot
        # Debounce updates from text inputs to avoid heavy re-computation
        # on every keystroke which can freeze the GUI when fitters run.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self.apply_thresholds)

        def _schedule_apply(*_args, **_kwargs):
            # restart the timer (300 ms) on each change; only when typing stops
            try:
                self._apply_timer.start(300)
            except Exception:
                # fallback: call immediately if timer unavailable
                try:
                    self.apply_thresholds()
                except Exception:
                    pass

        for widget in (
            self.x_increase_input,
            self.x_decrease_input,
            self.indiff_input,
            self.indiff25_input,
            self.indiff75_input,
            self.peak_location_input,
            self.left_indiff_input,
            self.right_indiff_input,
        ):
            # schedule apply on text changes (debounced)
            widget.textChanged.connect(_schedule_apply)
            # allow only numeric input (floating); validator range will be updated
            try:
                validator = QDoubleValidator(-1e12, 1e12, 12, self)
                # accept intermediate input so user can type negative sign or dot
                validator.setNotation(QDoubleValidator.StandardNotation)
                widget.setValidator(validator)
            except Exception:
                pass

        # helper to set a lineedit validator range when attribute bounds are known
        def set_lineedit_range(le, minimum, maximum):
            try:
                v = QDoubleValidator(float(minimum), float(maximum), 12, self)
                v.setNotation(QDoubleValidator.StandardNotation)
                le.setValidator(v)
            except Exception:
                try:
                    # fallback to permissive validator
                    le.setValidator(QDoubleValidator(-1e12, 1e12, 12, self))
                except Exception:
                    pass
        self._set_lineedit_range = set_lineedit_range

        # Polynomial degree slider
        self.degree_slider = QSlider(Qt.Horizontal)
        self.degree_slider.setMinimum(1)
        self.degree_slider.setMaximum(10)
        self.degree_slider.setValue(2)
        self.degree_label = QLabel("Polynomial Degree: 2")

        # Fit type selector (Piecewise linear by default)
        self.fit_type_selector = QComboBox()
        self.fit_type_selector.addItems([
            "Piecewise Linear",
            "Polynomial",
            "Monotone Spline (PCHIP)",
            "Gaussian",
            "Sigmoid",
        ])
        self.fit_type_selector.setCurrentIndex(0)
        self.fit_type_selector.currentIndexChanged.connect(self.on_fit_type_changed)

        # Parameter controls (created once; visibility toggled by fit type)
        # We'll create a dictionary of QLabel + QSlider + value QLabel for common params
        self.param_controls = {}
        SLIDER_STEPS = 1000
        self._SLIDER_STEPS = SLIDER_STEPS
        def make_param(name, minimum=-1e3, maximum=1e3, default=0.0):
            lbl = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, SLIDER_STEPS)
            rng = float(maximum) - float(minimum)
            scale = rng / SLIDER_STEPS if SLIDER_STEPS > 0 and rng != 0 else 1.0
            # compute initial slider position
            try:
                init = int((float(default) - float(minimum)) / scale) if scale != 0 else 0
            except Exception:
                init = 0
            slider.setValue(max(0, min(SLIDER_STEPS, init)))
            val_lbl = QLabel(f"{default:.4g}")
            # hide by default
            lbl.setVisible(False); slider.setVisible(False); val_lbl.setVisible(False)

            def on_slide(v, name=name, minv=minimum, sc=scale, vl=val_lbl):
                try:
                    fv = float(minv) + float(v) * sc
                    vl.setText(f"{fv:.4g}")
                except Exception:
                    vl.setText(str(v))
                # update plot on change
                try:
                    self.update_plot()
                except Exception:
                    pass

            slider.valueChanged.connect(on_slide)
            self.param_controls[name] = (lbl, slider, val_lbl, float(minimum), float(maximum), float(scale))
            return lbl, slider, val_lbl

        # Gaussian: amplitude, mu, sigma, offset
        make_param('amplitude', -1000, 1000, 1.0)
        make_param('mu', -1000, 1000, 0.0)
        make_param('sigma', 0.01, 1000, 1.0)
        make_param('offset', -1000, 1000, 0.0)
        # Sigmoid: L, k, x0, offset (offset named offset_sig internally to avoid collision)
        make_param('L', -1000, 1000, 1.0)
        # k will be interpreted on a log scale (user-facing range -3..3)
        make_param('k', -3.0, 3.0, -1.0)
        make_param('x0', -1000, 1000, 0.0)
        make_param('offset_sig', -1000, 1000, 0.0)
    # Exponential/Logarithmic removed per user request — parameters not exposed

        # Navigation buttons
        self.prev_button = QPushButton("Previous Attribute")
        self.next_button = QPushButton("Next Attribute")
        self.prev_button.clicked.connect(self.prev_attribute)
        self.next_button.clicked.connect(self.next_attribute)

        # Matplotlib canvas for plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Add widgets to layout
        self.layout.addWidget(self.file_label)
        # place the two CSV buttons side-by-side
        top_btns = QHBoxLayout()
        top_btns.addWidget(self.upload_button)
        self.layout.addLayout(top_btns)
        self.layout.addWidget(self.attr_label)
        # Add new UI controls for monotonicity and thresholds
        self.layout.addWidget(QLabel("Select behavior:"))
        self.layout.addWidget(self.mono_selector)
        self.layout.addWidget(self.shape_selector)
        # indifference / peak widgets (shown depending on selection)
        self.layout.addWidget(QLabel("Indifference / peak inputs:"))
        self.layout.addWidget(self.indiff_input)
        self.layout.addWidget(self.indiff25_input)
        self.layout.addWidget(self.indiff75_input)
        self.layout.addWidget(self.peak_location_input)
        self.layout.addWidget(self.left_indiff_input)
        self.layout.addWidget(self.right_indiff_input)
        self.layout.addWidget(QLabel("Thresholds:"))
        self.layout.addWidget(self.x_increase_input)
        self.layout.addWidget(self.x_decrease_input)
        # Move fit controls to the bottom (above the plot)
        fit_container_label = QLabel("Fit type:")
        self.layout.addWidget(fit_container_label)
        self.layout.addWidget(self.fit_type_selector)
        # add param controls in a compact horizontal layout
        params_layout = QHBoxLayout()
        for name, tpl in self.param_controls.items():
            lbl, slider, val_lbl, mn, mx, sc = tpl
            params_layout.addWidget(lbl)
            params_layout.addWidget(slider)
            params_layout.addWidget(val_lbl)
        self.layout.addLayout(params_layout)
        # degree controls
        self.degree_label.setVisible(False)
        self.degree_slider.setVisible(False)
        self.layout.addWidget(self.degree_label)
        self.layout.addWidget(self.degree_slider)

        # finally the plot canvas
        self.layout.addWidget(self.canvas)

        # Navigation buttons layout
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        self.layout.addLayout(nav_layout)

        # Connect slider
        self.degree_slider.valueChanged.connect(self.update_degree_label)

        # Ensure indifference/peak inputs visibility matches initial selection
        self.on_mono_changed(self.mono_selector.currentIndex())

        # Initial UI state
        self.update_ui()

    def upload_csv(self):
        """Open a file dialog to upload a CSV."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.process.load_data(file_path)
                self.file_label.setText(f"Loaded: {file_path}")
                # after loading, try to load saved state for current attribute
                # Ask user if they want to load an existing `value_functions.csv` or start new
                # First, look for a candidate in the same folder
                folder = os.path.dirname(file_path)
                candidate = os.path.join(folder, 'value_functions.csv')
                try:
                    if os.path.exists(candidate):
                        resp = QMessageBox.question(
                            self,
                            "Load existing value functions?",
                            f"Found 'value_functions.csv' in the same folder. Load it?\n\n{candidate}",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if resp == QMessageBox.Yes:
                            # load the candidate without asking again
                            self.load_elicited_csv(candidate)
                        else:
                            # user chose not to load; initialize UI normally
                            self.update_ui()
                    else:
                        # ask whether to browse for an existing file or start new
                        resp = QMessageBox.question(
                            self,
                            "Load value functions?",
                            "Do you want to load an existing 'value_functions.csv' file?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        if resp == QMessageBox.Yes:
                            # open file dialog to pick the file
                            self.load_elicited_csv(None)
                        else:
                            self.update_ui()
                except Exception:
                    # on any error during the prompt/load, fall back to normal UI init
                    self.update_ui()
            except Exception as e:
                self.file_label.setText(f"Error: {e}")

    def load_elicited_csv(self, file_path=None):
        """Optionally load a CSV that contains elicited points/meta (e.g. `value_functions.csv`).

        The file must contain a `name` column to match attributes in the currently
        loaded CSV. Matching rows will be merged into `self.process.df` in the
        columns `elicited_points`, `value_function`, `elicitation_meta`.
        """
        if self.process.df is None:
            self.file_label.setText("Load attribute CSV first before importing elicited values")
            return
        # If not provided, ask the user to pick a file
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Elicited CSV", "", "CSV Files (*.csv)")
            if not file_path:
                return
        try:
            import pandas as pd
            ef = pd.read_csv(file_path)
        except Exception as e:
            self.file_label.setText(f"Failed to read elicited CSV: {e}")
            return

        # Need a name column to match
        if 'name' not in ef.columns:
            self.file_label.setText("Elicited CSV must contain a 'name' column")
            return

        # Merge rows by name into process.df
        try:
            # ensure our DF has an index we can use
            for _, row in ef.iterrows():
                nm = row.get('name')
                if pd.isna(nm):
                    continue
                # find matching rows in process.df
                try:
                    matches = self.process.df.index[self.process.df['name'] == nm].tolist()
                except Exception:
                    # fallback: use boolean mask
                    matches = list(self.process.df[self.process.df['name'] == nm].index)
                for idx in matches:
                    for col in ('elicited_points', 'value_function', 'elicitation_meta'):
                        if col in ef.columns and not pd.isna(row.get(col)):
                            try:
                                self.process.df.at[idx, col] = row.get(col)
                            except Exception:
                                pass
            # persist merged DF back to process (in-memory) and refresh UI
            self.file_label.setText(f"Merged elicited CSV: {file_path}")
            # refresh UI to apply any saved state for current attribute
            self.update_ui()
        except Exception as e:
            self.file_label.setText(f"Error merging elicited CSV: {e}")

    def collect_meta_from_ui(self):
        """Collect elicitation options from the UI into a dict suitable for saving."""
        # Validate inputs before collecting meta. Raise ValueError on invalid input
        if self.process.df is None:
            raise ValueError("No CSV loaded")

        # get attribute bounds for validation
        try:
            attr = self.process.get_current_attribute()
            amin = float(attr['min']); amax = float(attr['max'])
        except Exception:
            amin = None; amax = None

        def parse_optional_float(w, name):
            txt = w.text().strip()
            if not txt:
                return None
            try:
                v = float(txt)
            except Exception:
                raise ValueError(f"Invalid number for {name}: '{txt}'")
            if amin is not None and amax is not None:
                if not (amin <= v <= amax):
                    raise ValueError(f"{name} value {v} out of range [{amin},{amax}]")
            return v

        meta = {}
        meta['mode'] = self.mono_selector.currentText()
        meta['shape'] = self.shape_selector.currentText()

        # thresholds (use empty as None)
        lower = None; upper = None
        if self.x_decrease_input.text().strip():
            lower = parse_optional_float(self.x_decrease_input, 'lower_threshold')
        if self.x_increase_input.text().strip():
            upper = parse_optional_float(self.x_increase_input, 'upper_threshold')
        # ensure thresholds ordering if both present
        if (lower is not None) and (upper is not None) and (lower > upper):
            raise ValueError(f"lower_threshold ({lower}) must be <= upper_threshold ({upper})")
        meta['lower_threshold'] = lower
        meta['upper_threshold'] = upper

        # indifference points
        x0 = parse_optional_float(self.indiff_input, 'x0') if self.indiff_input.text().strip() else None
        x025 = parse_optional_float(self.indiff25_input, 'x025') if self.indiff25_input.text().strip() else None
        x075 = parse_optional_float(self.indiff75_input, 'x075') if self.indiff75_input.text().strip() else None
        # ordering checks when present. Behavior depends on monotonicity direction.
        increasing = True
        try:
            if meta.get('mode') == 'Monotonic':
                sh = self.shape_selector.currentText()
                if 'Decreasing' in sh:
                    increasing = False
        except Exception:
            increasing = True

        if increasing:
            # require x025 < x0 < x075 when applicable
            if x025 is not None and x0 is not None and not (x025 < x0):
                raise ValueError("Require x0.25 < x0.5")
            if x0 is not None and x075 is not None and not (x0 < x075):
                raise ValueError("Require x0.5 < x0.75")
            if x025 is not None and x075 is not None and not (x025 < x075):
                raise ValueError("Require x0.25 < x0.75")
        else:
            # decreasing: reversed order expected (x0.75 < x0.5 < x0.25)
            if x075 is not None and x0 is not None and not (x075 < x0):
                raise ValueError("Require x0.75 < x0.5 for decreasing mode")
            if x0 is not None and x025 is not None and not (x0 < x025):
                raise ValueError("Require x0.5 < x0.25 for decreasing mode")
            if x075 is not None and x025 is not None and not (x075 < x025):
                raise ValueError("Require x0.75 < x0.25 for decreasing mode")

        meta['x0'] = x0
        meta['x025'] = x025
        meta['x075'] = x075

        meta['peak_location'] = parse_optional_float(self.peak_location_input, 'peak_location') if self.peak_location_input.text().strip() else None
        meta['left_indiff'] = parse_optional_float(self.left_indiff_input, 'left_indiff') if self.left_indiff_input.text().strip() else None
        meta['right_indiff'] = parse_optional_float(self.right_indiff_input, 'right_indiff') if self.right_indiff_input.text().strip() else None

        # fit settings
        try:
            meta['fit_type'] = self.fit_type_selector.currentText()
        except Exception:
            meta['fit_type'] = 'Piecewise Linear'

        # collect visible param sliders into fit_params (map internal names to fit param names)
        name_map = {
            'offset_sig': 'offset'
        }
        fit_params = {}
        for iname, tpl in getattr(self, 'param_controls', {}).items():
            lbl, slider, val_lbl, mn, mx, sc = tpl
            if slider.isVisible():
                try:
                    fv = float(mn) + float(slider.value()) * float(sc)
                    pname = name_map.get(iname, iname)
                    fit_params[pname] = float(fv)
                except Exception:
                    pass
        # ensure Gaussian fixed params are saved even when sliders hidden
        try:
            if meta.get('fit_type') == 'Gaussian':
                fit_params.setdefault('amplitude', 1.0)
                fit_params.setdefault('offset', 0.0)
        except Exception:
            pass

        meta['fit_params'] = fit_params
        return meta

    def apply_saved_state_to_ui(self, state):
        """Apply saved points/meta (from process.get_saved_state_for_current_attribute()) to UI and process.

        state is a dict with keys 'points', 'value_function', 'meta'
        """
        if not state:
            return
        pts = state.get('points')
        meta = state.get('meta')
        if pts:
            # pts expected as list of [x,y] pairs
            try:
                self.process.points = [(float(p[0]), float(p[1])) for p in pts]
            except Exception:
                self.process.points = []

        if meta:
            # apply UI fields where present
            mode = meta.get('mode')
            if mode:
                try:
                    idx = 0 if mode == 'Monotonic' else 1
                    self.mono_selector.setCurrentIndex(idx)
                    # Ensure the shape selector options are in the correct semantic set
                    try:
                        self.on_mono_changed(self.mono_selector.currentIndex())
                    except Exception:
                        pass
                except Exception:
                    pass
            shape = meta.get('shape')
            if shape:
                try:
                    # set shape selector if matches. Shape strings may be 'Concave (∩)',
                    # 'Convex (∪)' for non-monotonic or 'Increasing'/'Decreasing' for monotonic.
                    # Use findText to handle the currently populated items.
                    # prefer exact match, fallback to substring checks
                    idx = self.shape_selector.findText(shape)
                    if idx is not None and idx >= 0:
                        self.shape_selector.setCurrentIndex(idx)
                    else:
                        if 'Concave' in shape:
                            self.shape_selector.setCurrentIndex(0)
                        elif 'Convex' in shape:
                            self.shape_selector.setCurrentIndex(1)
                        elif 'Increasing' in shape:
                            # for monotonic mode, shape_selector contains 'Increasing'/'Decreasing'
                            try:
                                idx2 = self.shape_selector.findText('Increasing')
                                if idx2 is not None and idx2 >= 0:
                                    self.shape_selector.setCurrentIndex(idx2)
                            except Exception:
                                pass
                        elif 'Decreasing' in shape:
                            try:
                                idx2 = self.shape_selector.findText('Decreasing')
                                if idx2 is not None and idx2 >= 0:
                                    self.shape_selector.setCurrentIndex(idx2)
                            except Exception:
                                pass
                except Exception:
                    pass
            # thresholds
            if meta.get('lower_threshold') is not None:
                self.x_decrease_input.setText(str(meta.get('lower_threshold')))
            if meta.get('upper_threshold') is not None:
                self.x_increase_input.setText(str(meta.get('upper_threshold')))
            # indifference points
            if meta.get('x0') is not None:
                self.indiff_input.setText(str(meta.get('x0')))
            if meta.get('x025') is not None:
                self.indiff25_input.setText(str(meta.get('x025')))
            if meta.get('x075') is not None:
                self.indiff75_input.setText(str(meta.get('x075')))
            if meta.get('peak_location') is not None:
                self.peak_location_input.setText(str(meta.get('peak_location')))
            if meta.get('left_indiff') is not None:
                self.left_indiff_input.setText(str(meta.get('left_indiff')))
            if meta.get('right_indiff') is not None:
                self.right_indiff_input.setText(str(meta.get('right_indiff')))
            # also set process threshold/tail metadata so plotting can use it
            try:
                if meta.get('lower_threshold') is not None:
                    self.process.lower_threshold = float(meta.get('lower_threshold'))
                if meta.get('upper_threshold') is not None:
                    self.process.upper_threshold = float(meta.get('upper_threshold'))
                # set tail defaults from meta if present, else infer from mode/shape
                if meta.get('left_tail') is not None:
                    self.process.left_tail_value = float(meta.get('left_tail'))
                if meta.get('right_tail') is not None:
                    self.process.right_tail_value = float(meta.get('right_tail'))
                # infer if missing
                if self.process.left_tail_value is None or self.process.right_tail_value is None:
                    if self.mono_selector.currentText() == 'Monotonic':
                        # determine increasing / decreasing
                        sh = self.shape_selector.currentText()
                        try:
                            if 'Decreasing' in sh:
                                self.process.left_tail_value = 1.0
                                self.process.right_tail_value = 0.0
                            else:
                                self.process.left_tail_value = 0.0
                                self.process.right_tail_value = 1.0
                        except Exception:
                            self.process.left_tail_value = 0.0
                            self.process.right_tail_value = 1.0
                    else:
                        # concave -> 0, convex -> 1
                        if 'Concave' in self.shape_selector.currentText():
                            self.process.left_tail_value = 0.0
                            self.process.right_tail_value = 0.0
                        else:
                            self.process.left_tail_value = 1.0
                            self.process.right_tail_value = 1.0
            except Exception:
                pass

        # restore fit type and slider parameters if present in meta
        if meta:
            try:
                if meta.get('fit_type'):
                    # set fit type and update visible controls
                    ft = meta.get('fit_type')
                    idx = 0
                    try:
                        idx = self.fit_type_selector.findText(ft)
                    except Exception:
                        idx = 0
                    if idx is None or idx < 0:
                        idx = 0
                    self.fit_type_selector.setCurrentIndex(idx)
                    # ensure UI reflects this
                    try:
                        self.on_fit_type_changed(idx)
                    except Exception:
                        pass
                # params
                fitp = meta.get('fit_params')
                if fitp and isinstance(fitp, dict):
                    # map fit param names back to internal slider names
                    rev_map = {'offset': 'offset_sig'}
                    # also L,k,x0 etc map directly
                    for pname, pval in fitp.items():
                        iname = rev_map.get(pname, pname)
                        if iname in self.param_controls:
                            lbl, slider, val_lbl, mn, mx, sc = self.param_controls[iname]
                            try:
                                slider.blockSignals(True)
                                pos = int((float(pval) - float(mn)) / float(sc)) if float(sc) != 0 else 0
                                pos = max(0, min(slider.maximum(), pos))
                                slider.setValue(pos)
                                val_lbl.setText(f"{float(pval):.4g}")
                            finally:
                                slider.blockSignals(False)
            except Exception:
                pass

    # manual single-point add removed: use thresholds/shape controls instead

    def on_mono_changed(self, idx):
        """Show/hide shape selector depending on monotonicity choice."""
        is_non_mono = (self.mono_selector.currentText() == "Non-monotonic")
        # Always show the shape selector, but change its semantics depending on mode.
        # For Monotonic: use Increasing / Decreasing
        # For Non-monotonic: use Concave (∩) / Convex (∪)
        try:
            current = self.shape_selector.currentText()
        except Exception:
            current = None
        try:
            self.shape_selector.blockSignals(True)
            self.shape_selector.clear()
            if is_non_mono:
                self.shape_selector.addItems(["Concave (∩)", "Convex (∪)"])
                # try to preserve previous selection if it matches
                if current and ("Concave" in current or "Convex" in current):
                    # set to previous index if possible
                    try:
                        idx = 0 if "Concave" in current else 1
                        self.shape_selector.setCurrentIndex(idx)
                    except Exception:
                        pass
            else:
                self.shape_selector.addItems(["Increasing", "Decreasing"])
                if current and ("Increasing" in current or "Decreasing" in current):
                    try:
                        idx = 0 if "Increasing" in current else 1
                        self.shape_selector.setCurrentIndex(idx)
                    except Exception:
                        pass
        finally:
            try:
                self.shape_selector.blockSignals(False)
            except Exception:
                pass
        self.shape_selector.setVisible(True)
        # Enable Gaussian only for non-monotonic (peaks/valleys)
        try:
            gi = self.fit_type_selector.findText('Gaussian')
            if gi is not None and gi >= 0:
                try:
                    item = self.fit_type_selector.model().item(gi)
                    if item is not None:
                        item.setEnabled(is_non_mono)
                except Exception:
                    # fallback: if disabling, ensure selection is not Gaussian
                    pass
            # if switched to monotonic while Gaussian is selected, choose a safe default
            if not is_non_mono and self.fit_type_selector.currentText() == 'Gaussian':
                self.fit_type_selector.setCurrentIndex(0)
        except Exception:
            pass
        # Show/hide indifference/peak inputs
        self.indiff_input.setVisible(not is_non_mono)
        # also show the 0.25/0.75 inputs for monotonic
        self.indiff25_input.setVisible(not is_non_mono)
        self.indiff75_input.setVisible(not is_non_mono)
        self.peak_location_input.setVisible(is_non_mono)
        self.left_indiff_input.setVisible(is_non_mono)
        self.right_indiff_input.setVisible(is_non_mono)
        # Immediately apply defaults/changes so plot updates without pressing Apply
        try:
            self.apply_thresholds()
        except Exception:
            pass

    def on_shape_changed(self, idx):
        """When shape (concave/convex) changes, update defaults immediately.

        Concave -> tails = 0 at extremes; Convex -> tails = 1 at extremes.
        Calling apply_thresholds ensures the plot updates immediately.
        """
        try:
            # trigger apply to set tails and default points
            self.apply_thresholds()
        except Exception:
            pass

    def on_fit_type_changed(self, idx):
        """Adjust UI elements (degree slider, params label) depending on fit type."""
        ft = self.fit_type_selector.currentText()
        # Show degree slider only for Polynomial
        is_poly = (ft == 'Polynomial')
        self.degree_slider.setVisible(is_poly)
        self.degree_label.setVisible(is_poly)

        # Hide all param controls then show required ones
        for name, tpl in self.param_controls.items():
            lbl, slider, val_lbl, mn, mx, sc = tpl
            lbl.setVisible(False)
            slider.setVisible(False)
            val_lbl.setVisible(False)

        if ft == 'Gaussian':
            # expose only mu and sigma; amplitude fixed to 1.0 and offset fixed to 0.0
            for n in ('mu', 'sigma'):
                lbl, slider, val_lbl, *_ = self.param_controls[n]
                lbl.setVisible(True); slider.setVisible(True); val_lbl.setVisible(True)
        elif ft == 'Sigmoid':
            # only expose center (x0) and steepness (k); offsets and L are handled by the fitter
            for n in ('k', 'x0'):
                lbl, slider, val_lbl, *_ = self.param_controls[n]
                lbl.setVisible(True); slider.setVisible(True); val_lbl.setVisible(True)
        # Exponential and Logarithmic options removed — nothing to show here

        # Trigger a replot with the new selection
        try:
            self.update_plot()
        except Exception:
            pass

    def apply_thresholds(self):
        """Read threshold inputs, use defaults from attribute, and add points accordingly.

        Assumptions made:
        - For Monotonic: we add (x_decrease, 0) and (x_increase, 1).
        - For Non-monotonic Concave (∩): we add points to form a peak: (attr.min,0), (x_inc,1), (x_dec,0), (attr.max,0).
        - For Non-monotonic Convex (∪): valley: (attr.min,1), (x_inc,0), (x_dec,1), (attr.max,1).
        - If an input is empty or invalid we use attribute min/max as defaults.
        """
        if self.process.df is None:
            self.file_label.setText("Load a CSV first")
            return

        attr = self.process.get_current_attribute()
        try:
            amin = float(attr['min'])
            amax = float(attr['max'])
        except Exception:
            self.file_label.setText("Attribute min/max invalid")
            return

        # Read inputs or fallback; enforce numeric input and range
        def _parse_in_range(widget, default):
            txt = widget.text().strip()
            if not txt:
                return default, None
            try:
                val = float(txt)
            except Exception:
                return None, f"Invalid number: '{txt}'"
            if not (amin <= val <= amax):
                return None, f"Value {val} out of range [{amin}, {amax}]"
            return val, None

        x_inc, err = _parse_in_range(self.x_increase_input, amax)
        if err:
            self.file_label.setText(err)
            try:
                self.x_increase_input.setStyleSheet("background-color: #ffcccc")
            except Exception:
                pass
            return
        else:
            try:
                self.x_increase_input.setStyleSheet("")
            except Exception:
                pass

        x_dec, err = _parse_in_range(self.x_decrease_input, amin)
        if err:
            self.file_label.setText(err)
            try:
                self.x_decrease_input.setStyleSheet("background-color: #ffcccc")
            except Exception:
                pass
            return
        else:
            try:
                self.x_decrease_input.setStyleSheet("")
            except Exception:
                pass

        # Reset current points for the attribute
        self.process.points = []

        # set thresholds and tail values on the process for piecewise plotting/fit
        self.process.lower_threshold = x_dec
        self.process.upper_threshold = x_inc

        if self.mono_selector.currentText() == "Monotonic":
            # For monotonic behavior, allow Increasing or Decreasing chosen
            shape = self.shape_selector.currentText()
            increasing = True
            try:
                if 'Decreasing' in shape:
                    increasing = False
            except Exception:
                increasing = True
            if increasing:
                # left tail 0, right tail 1
                self.process.left_tail_value = 0.0
                self.process.right_tail_value = 1.0
                # Add lower (0) then upper (1)
                self.process.add_point(x_dec, 0.0)
                self.process.add_point(x_inc, 1.0)
            else:
                # decreasing: invert tails and point values
                self.process.left_tail_value = 1.0
                self.process.right_tail_value = 0.0
                self.process.add_point(x_dec, 1.0)
                self.process.add_point(x_inc, 0.0)
            # optional indifference point for monotonic
            # parse optional points and enforce ordering x025 < x0 < x075 when present
            x0 = x25 = x75 = None
            if self.indiff_input.text().strip():
                try:
                    x0_val = float(self.indiff_input.text())
                    if not (amin <= x0_val <= amax):
                        self.file_label.setText(f"x0 out of range [{amin},{amax}]")
                        self.indiff_input.setStyleSheet("background-color: #ffcccc")
                        return
                    x0 = x0_val
                    self.indiff_input.setStyleSheet("")
                    self.process.add_point(x0, 0.5)
                except Exception:
                    self.file_label.setText("Invalid x0 value")
                    self.indiff_input.setStyleSheet("background-color: #ffcccc")
                    return

            if self.indiff25_input.text().strip():
                try:
                    x25_val = float(self.indiff25_input.text())
                    if not (amin <= x25_val <= amax):
                        self.file_label.setText(f"x0.25 out of range [{amin},{amax}]")
                        self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                        return
                    x25 = x25_val
                    self.indiff25_input.setStyleSheet("")
                except Exception:
                    self.file_label.setText("Invalid x0.25 value")
                    self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                    return

            if self.indiff75_input.text().strip():
                try:
                    x75_val = float(self.indiff75_input.text())
                    if not (amin <= x75_val <= amax):
                        self.file_label.setText(f"x0.75 out of range [{amin},{amax}]")
                        self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                        return
                    x75 = x75_val
                    self.indiff75_input.setStyleSheet("")
                except Exception:
                    self.file_label.setText("Invalid x0.75 value")
                    self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                    return

            # ordering check if at least two of them present: require x25 < x0 < x75 when applicable
            try:
                vals = {'x25': x25, 'x0': x0, 'x75': x75}
                present_keys = [k for k, v in vals.items() if v is not None]
                if len(present_keys) >= 2:
                    # define expected order depending on increasing/decreasing
                    if increasing:
                        expected = ['x25', 'x0', 'x75']
                        err_msg = "Require x0.25 < x0.5 < x0.75"
                    else:
                        expected = ['x75', 'x0', 'x25']
                        err_msg = "Require x0.75 < x0.5 < x0.25"

                    # all three present: enforce full ordering
                    if all(vals[k] is not None for k in expected):
                        if not (vals[expected[0]] < vals[expected[1]] < vals[expected[2]]):
                            self.file_label.setText(err_msg)
                            self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                            self.indiff_input.setStyleSheet("background-color: #ffcccc")
                            self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                            return
                    else:
                        # two present: ensure their relative order matches expected ordering
                        if len(present_keys) == 2:
                            k1, k2 = present_keys[0], present_keys[1]
                            try:
                                idx1 = expected.index(k1)
                                idx2 = expected.index(k2)
                            except Exception:
                                idx1 = idx2 = 0
                            v1 = vals[k1]; v2 = vals[k2]
                            if idx1 < idx2:
                                if not (v1 < v2):
                                    self.file_label.setText("Indifference points ordering invalid")
                                    if k1 == 'x25': self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                                    if k1 == 'x0': self.indiff_input.setStyleSheet("background-color: #ffcccc")
                                    if k1 == 'x75': self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x25': self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x0': self.indiff_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x75': self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                                    return
                            else:
                                if not (v1 > v2):
                                    self.file_label.setText("Indifference points ordering invalid")
                                    if k1 == 'x25': self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                                    if k1 == 'x0': self.indiff_input.setStyleSheet("background-color: #ffcccc")
                                    if k1 == 'x75': self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x25': self.indiff25_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x0': self.indiff_input.setStyleSheet("background-color: #ffcccc")
                                    if k2 == 'x75': self.indiff75_input.setStyleSheet("background-color: #ffcccc")
                                    return
            except Exception:
                pass
            # Add parsed x25/x75 points following original ordering rules
            try:
                if x25 is not None:
                    # if x0 exists enforce ordering (min <= x25 <= x0) for increasing, else (x0 <= x25 <= max)
                    if x0 is not None:
                        if increasing:
                            if amin <= x25 <= x0:
                                self.process.add_point(x25, 0.25)
                        else:
                            if x0 <= x25 <= amax:
                                self.process.add_point(x25, 0.25)
                    else:
                        if amin <= x25 <= amax:
                            self.process.add_point(x25, 0.25)
                if x75 is not None:
                    if x0 is not None:
                        if increasing:
                            if x0 <= x75 <= amax:
                                self.process.add_point(x75, 0.75)
                        else:
                            if amin <= x75 <= x0:
                                self.process.add_point(x75, 0.75)
                    else:
                        if amin <= x75 <= amax:
                            self.process.add_point(x75, 0.75)
            except Exception:
                pass
        else:
            shape = self.shape_selector.currentText()
            if shape.startswith("Concave"):
                # Peak (concave): tails 0, inner thresholds set to 0, peak (if any) at y=1
                self.process.left_tail_value = 0.0
                self.process.right_tail_value = 0.0
                self.process.add_point(amin, 0.0)
                self.process.add_point(x_inc, 0.0)
                self.process.add_point(x_dec, 0.0)
                self.process.add_point(amax, 0.0)
                # optional peak
                try:
                    if self.peak_location_input.text().strip():
                        px = float(self.peak_location_input.text())
                        if amin <= px <= amax:
                            self.process.add_point(px, 1.0)
                except Exception:
                    pass
                # left/right indifference points (must be within range and on correct sides of peak if provided)
                try:
                    if self.left_indiff_input.text().strip():
                        lx = float(self.left_indiff_input.text())
                        if not (amin <= lx <= amax):
                            self.file_label.setText(f"Left indifference out of range [{amin},{amax}]")
                            self.left_indiff_input.setStyleSheet("background-color: #ffcccc")
                            return
                        try:
                            if 'px' in locals() and not (lx <= px):
                                self.file_label.setText("Left indifference must not be larger than peak")
                                self.left_indiff_input.setStyleSheet("background-color: #ffcccc")
                                return
                        except Exception:
                            pass
                        self.left_indiff_input.setStyleSheet("")
                        self.process.add_point(lx, 0.5)
                except Exception:
                    pass
                try:
                    if self.right_indiff_input.text().strip():
                        rx = float(self.right_indiff_input.text())
                        if not (amin <= rx <= amax):
                            self.file_label.setText(f"Right indifference out of range [{amin},{amax}]")
                            self.right_indiff_input.setStyleSheet("background-color: #ffcccc")
                            return
                        try:
                            if 'px' in locals() and not (rx >= px):
                                self.file_label.setText("Right indifference must not be smaller than peak")
                                self.right_indiff_input.setStyleSheet("background-color: #ffcccc")
                                return
                        except Exception:
                            pass
                        self.right_indiff_input.setStyleSheet("")
                        self.process.add_point(rx, 0.5)
                except Exception:
                    pass
            else:
                # Convex (valley): tails 1, inner thresholds set to 1, valley (if any) at y=0
                self.process.left_tail_value = 1.0
                self.process.right_tail_value = 1.0
                self.process.add_point(amin, 1.0)
                self.process.add_point(x_inc, 1.0)
                self.process.add_point(x_dec, 1.0)
                self.process.add_point(amax, 1.0)
                # optional valley
                try:
                    if self.peak_location_input.text().strip():
                        px = float(self.peak_location_input.text())
                        if amin <= px <= amax:
                            self.process.add_point(px, 0.0)
                except Exception:
                    pass
                # left/right indifference points (must be within range and on correct sides of valley if provided)
                try:
                    if self.left_indiff_input.text().strip():
                        lx = float(self.left_indiff_input.text())
                        if not (amin <= lx <= amax):
                            self.file_label.setText(f"Left indifference out of range [{amin},{amax}]")
                            self.left_indiff_input.setStyleSheet("background-color: #ffcccc")
                            return
                        try:
                            if 'px' in locals() and not (lx <= px):
                                self.file_label.setText("Left indifference must not be larger than valley")
                                self.left_indiff_input.setStyleSheet("background-color: #ffcccc")
                                return
                        except Exception:
                            pass
                        self.left_indiff_input.setStyleSheet("")
                        self.process.add_point(lx, 0.5)
                except Exception:
                    pass
                try:
                    if self.right_indiff_input.text().strip():
                        rx = float(self.right_indiff_input.text())
                        if not (amin <= rx <= amax):
                            self.file_label.setText(f"Right indifference out of range [{amin},{amax}]")
                            self.right_indiff_input.setStyleSheet("background-color: #ffcccc")
                            return
                        try:
                            if 'px' in locals() and not (rx >= px):
                                self.file_label.setText("Right indifference must not be smaller than valley")
                                self.right_indiff_input.setStyleSheet("background-color: #ffcccc")
                                return
                        except Exception:
                            pass
                        self.right_indiff_input.setStyleSheet("")
                        self.process.add_point(rx, 0.5)
                except Exception:
                    pass

        # Deduplicate points by x so that later additions (endpoints) override earlier ones.
        if self.process.points:
            seen = {}
            # preserve last occurrence for each x
            for (px, py) in self.process.points:
                seen[float(px)] = float(py)
            # store points sorted by x to make plotting and interpolation natural
            self.process.points = [(x, seen[x]) for x in sorted(seen.keys())]

        self.update_plot()

    def update_plot(self):
        """Update the plot with current points and fitted curve."""
        # Prepare data for plotting
        x_fit, y_fit = None, None
        try:
            degree = self.degree_slider.value()
            # parse fit type and params from UI (use visible sliders)
            fit_type = self.fit_type_selector.currentText() if hasattr(self, 'fit_type_selector') else 'Piecewise Linear'
            params = {}
            # mapping internal control names to fit param names when needed
            name_map = {
                'offset_sig': 'offset',
                'la': 'a', 'lb': 'b', 'lc': 'c', 'ld': 'd'
            }
            # read all visible param sliders
            for iname, tpl in getattr(self, 'param_controls', {}).items():
                lbl, slider, val_lbl, mn, mx, sc = tpl
                if slider.isVisible():
                    try:
                        fv = float(mn) + float(slider.value()) * float(sc)
                        pname = name_map.get(iname, iname)
                        params[pname] = float(fv)
                    except Exception:
                        pass

            # Pass monotonicity flags from UI into params so fitters know direction
            try:
                # default: not enforcing monotonic unless UI requests it
                if hasattr(self, 'mono_selector') and self.mono_selector.currentText() == 'Monotonic':
                    # shape_selector contains 'Increasing' or 'Decreasing' in monotonic mode
                    sh = self.shape_selector.currentText() if hasattr(self, 'shape_selector') else ''
                    increasing_flag = True
                    try:
                        if 'Decreasing' in sh:
                            increasing_flag = False
                    except Exception:
                        increasing_flag = True
                    params['increasing'] = bool(increasing_flag)
                    # signal to polynomial fitter that monotonic constraint should be applied
                    params['monotonic'] = True
                else:
                    params.setdefault('monotonic', False)
            except Exception:
                # be tolerant: do not break plotting if UI not available
                pass

            # compute attribute range to adjust relative parameters (e.g. sigmoid k, exp/log b)
            try:
                attr = self.process.get_current_attribute()
                xmin = float(attr['min']); xmax = float(attr['max'])
                rng = max(1e-6, xmax - xmin)
            except Exception:
                rng = 1.0

            # For sigmoid, interpret 'k' slider as a log-scale relative steepness.
            # Slider stores a logical value (e.g. -3..3). We map it to k = sign * 10**abs(raw_k) / rng
            if fit_type == 'Sigmoid' and 'k' in params:
                try:
                    raw_k = float(params['k'])
                    sign = 1.0 if raw_k >= 0 else -1.0
                    k_eff = sign * (10.0 ** (abs(raw_k))) / float(rng)
                    params['k'] = float(k_eff)
                except Exception:
                    pass

            # Exponential/Logarithmic fits removed — no special parameter transforms

            # delegate fitting to process (new fit_curve method)
            fit_result = None
            if hasattr(self.process, 'fit_curve'):
                # For Gaussian, force amplitude=1 and offset=0 (user not allowed to change)
                if fit_type == 'Gaussian':
                    try:
                        params['amplitude'] = 1.0
                    except Exception:
                        params.update({'amplitude': 1.0})
                    try:
                        params['offset'] = 0.0
                    except Exception:
                        params.update({'offset': 0.0})
                fit_result = self.process.fit_curve(fit_type=fit_type, degree=degree, params=params)
            else:
                # fallback to original polynomial fit
                fit_result = self.process.fit_polynomial(degree)

            # Unpack fit_result which may be (x_fit, y_fit) or (x_fit, y_fit, fit_params)
            fit_params_result = {}
            if fit_result is None:
                x_fit, y_fit = None, None
            elif isinstance(fit_result, tuple) and len(fit_result) == 3:
                x_fit, y_fit, fit_params_result = fit_result
            elif isinstance(fit_result, tuple) and len(fit_result) == 2:
                x_fit, y_fit = fit_result
            else:
                # unknown shape
                x_fit, y_fit = None, None
        except Exception:
            # let the plotter handle missing fit gracefully
            x_fit, y_fit = None, None

        # If the selected fit is Sigmoid and no fitter result is available,
        # compute the sigmoid directly from the visible sliders (like draft2.py):
        # y = 1 / (1 + exp(-k * (x - x0))) with k interpreted on a log scale.
        try:
            if (fit_type == 'Sigmoid') and (x_fit is None):
                # get attribute bounds for x grid
                try:
                    attr = self.process.get_current_attribute()
                    xmin = float(attr['min']); xmax = float(attr['max'])
                except Exception:
                    xmin, xmax = 0.0, 1.0
                x_test = np.linspace(xmin, xmax, 200)
                # x0 from params if present, else midpoint
                x0 = params.get('x0', (xmin + xmax) / 2.0)
                # raw_k may already have been transformed into an effective k in params
                if 'k' in params:
                    # params['k'] contains the effective k after the earlier transform
                    k_eff = float(params['k'])
                else:
                    # fallback: small default steepness relative to range
                    k_eff = 1.0 / max(1e-6, (xmax - xmin))
                y_test = 1.0 / (1.0 + np.exp(-k_eff * (x_test - float(x0))))
                # enforce piecewise constant tails based on process thresholds/tails
                low_thr = getattr(self.process, 'lower_threshold', xmin)
                high_thr = getattr(self.process, 'upper_threshold', xmax)
                left_tail = getattr(self.process, 'left_tail_value', 0.0)
                right_tail = getattr(self.process, 'right_tail_value', 1.0)
                try:
                    y_test[x_test <= float(low_thr)] = float(left_tail)
                except Exception:
                    pass
                try:
                    y_test[x_test >= float(high_thr)] = float(right_tail)
                except Exception:
                    pass
                y_test = np.clip(y_test, 0.0, 1.0)
                x_fit, y_fit = x_test, y_test
        except Exception:
            # if anything goes wrong here, fall back to fitter result (if any)
            pass

        xmin = xmax = None
        title = None
        if self.process.df is not None:
            try:
                attr = self.process.get_current_attribute()
                xmin = attr['min']
                xmax = attr['max']
                # build plot title from group and attribute name if available
                try:
                    group = attr.get('group') if hasattr(attr, 'get') else None
                    name = attr.get('name') if hasattr(attr, 'get') else None
                    if group is None:
                        # fallback: try attribute access by key
                        group = attr['group'] if 'group' in attr.index else None
                    if name is None:
                        name = attr['name'] if 'name' in attr.index else None
                    if group is not None and name is not None:
                        title = f"[{group}] - {name}"
                    elif name is not None:
                        title = f"{name}"
                except Exception:
                    title = None
            except Exception:
                xmin = xmax = None

            # If the fit was automatic (user did not supply params) and the fitter returned
            # parameter estimates, populate visible sliders so the user sees the "ideal" fit.
            try:
                if (not params) and fit_params_result and isinstance(fit_params_result, dict):
                    # map fit param names back to internal slider names
                    rev_map = {'offset': 'offset_sig', 'a': 'la', 'b': 'lb', 'c': 'lc', 'd': 'ld'}
                    for pname, pval in fit_params_result.items():
                        # special-case sigmoid k: convert effective k back to slider raw (log) value
                        try:
                            if pname == 'k' and self.fit_type_selector.currentText() == 'Sigmoid':
                                # need attribute range
                                try:
                                    attr = self.process.get_current_attribute()
                                    xmin = float(attr['min']); xmax = float(attr['max'])
                                    rng = max(1e-6, xmax - xmin)
                                except Exception:
                                    rng = 1.0
                                kev = float(pval)
                                sign = 1.0 if kev >= 0 else -1.0
                                rawk = 0.0
                                try:
                                    rawk = sign * np.log10(max(1e-12, abs(kev * rng)))
                                except Exception:
                                    rawk = 0.0
                                iname = 'k'
                                pval_to_use = float(rawk)
                            else:
                                iname = rev_map.get(pname, pname)
                                pval_to_use = float(pval)
                        except Exception:
                            continue
                        if iname in self.param_controls:
                            lbl, slider, val_lbl, mn, mx, sc = self.param_controls[iname]
                            try:
                                slider.blockSignals(True)
                                pos = int((float(pval_to_use) - float(mn)) / float(sc)) if float(sc) != 0 else 0
                                pos = max(0, min(slider.maximum(), pos))
                                slider.setValue(pos)
                                val_lbl.setText(f"{float(pval_to_use):.4g}")
                            finally:
                                slider.blockSignals(False)
            except Exception:
                pass

        # Delegate all plotting to the Plotter (keeps UI minimal)
        try:
            self.plotter.plot(
                self.ax,
                self.process.points,
                x_fit,
                y_fit,
                xmin,
                xmax,
                lower_threshold=getattr(self.process, 'lower_threshold', None),
                upper_threshold=getattr(self.process, 'upper_threshold', None),
                left_tail_value=getattr(self.process, 'left_tail_value', None),
                right_tail_value=getattr(self.process, 'right_tail_value', None),
                title=title,
            )
            self.canvas.draw()
        except Exception as e:
            # surface non-fatal plotting errors to the UI label
            self.file_label.setText(f"Plot error: {e}")

    def update_ui(self):
        """Update the UI based on the current state."""
        if self.process.df is not None:
            attr = self.process.get_current_attribute()
            self.attr_label.setText(
                f"Attribute: {attr['name']} (Range: {attr['min']} to {attr['max']})"
            )
            # update placeholders/defaults for threshold inputs
            try:
                amin = float(attr['min'])
                amax = float(attr['max'])
                self.x_increase_input.setPlaceholderText(f"X after which increasing not important (default = {amax})")
                self.x_decrease_input.setPlaceholderText(f"X after which decreasing not important (default = {amin})")
            except Exception:
                pass
                # update lineedit validators ranges when attribute bounds known
            try:
                if hasattr(self, '_set_lineedit_range'):
                    self._set_lineedit_range(self.x_increase_input, amin, amax)
                    self._set_lineedit_range(self.x_decrease_input, amin, amax)
                    self._set_lineedit_range(self.indiff_input, amin, amax)
                    self._set_lineedit_range(self.indiff25_input, amin, amax)
                    self._set_lineedit_range(self.indiff75_input, amin, amax)
                    self._set_lineedit_range(self.peak_location_input, amin, amax)
                    self._set_lineedit_range(self.left_indiff_input, amin, amax)
                    self._set_lineedit_range(self.right_indiff_input, amin, amax)
            except Exception:
                pass
                # update parameter slider ranges based on attribute span
            try:
                rng = max(1e-6, float(amax) - float(amin))
                mid = (float(amin) + float(amax)) / 2.0
                # mu and x0 should lie within attribute bounds
                self.set_param_range('mu', amin, amax, default=mid)
                self.set_param_range('x0', amin, amax, default=mid)
                # sigma: small positive to reasonable fraction of range
                sigma_min = max(rng / 1000.0, 1e-6)
                sigma_max = max(rng, 1.0)
                self.set_param_range('sigma', sigma_min, sigma_max, default=max(rng / 6.0, sigma_min))
                # amplitude / L and offsets should be in [0,1]
                self.set_param_range('amplitude', 0.0, 1.0, default=1.0)
                self.set_param_range('L', 0.0, 1.0, default=1.0)
                self.set_param_range('offset_sig', 0.0, 1.0, default=0.0)
                self.set_param_range('offset', 0.0, 1.0, default=0.0)
                # steepness k reasonable bounds (log-scale user slider)
                self.set_param_range('k', -3.0, 3.0, default=-1.0)
            except Exception:
                pass
            # If the CSV has saved state for this attribute, load it
            saved = self.process.get_saved_state_for_current_attribute()
            if saved and (saved.get('points') or saved.get('meta')):
                # apply saved points and meta to process/UI
                self.apply_saved_state_to_ui(saved)
                self.update_plot()
            else:
                # If no saved points defined yet for this attribute, initialize defaults
                if not self.process.points:
                    # apply_thresholds will use attribute defaults when inputs are empty
                    self.apply_thresholds()
                else:
                    self.update_plot()
        else:
            self.attr_label.setText("No attribute selected")
            self.ax.clear()
            self.canvas.draw()

    def reset_ui_fields(self):
        """Clear all input fields and reset process metadata to defaults.

        This ensures when navigating attributes the UI doesn't keep values from
        the previously viewed attribute. After calling this, either saved state
        should be applied or defaults initialized via apply_thresholds().
        """
        # Clear line edits
        for w in (
            self.x_increase_input,
            self.x_decrease_input,
            self.indiff_input,
            self.indiff25_input,
            self.indiff75_input,
            self.peak_location_input,
            self.left_indiff_input,
            self.right_indiff_input,
        ):
            try:
                w.blockSignals(True)
                w.setText("")
            finally:
                w.blockSignals(False)

        # Reset selectors
        try:
            self.mono_selector.blockSignals(True)
            self.mono_selector.setCurrentIndex(0)
        finally:
            self.mono_selector.blockSignals(False)
        # Ensure shape/mono visibility and semantics are consistent after reset
        try:
            # reset shape index but DO NOT explicitly hide it here; on_mono_changed will set visibility
            self.shape_selector.blockSignals(True)
            self.shape_selector.setCurrentIndex(0)
        finally:
            self.shape_selector.blockSignals(False)

        # Apply mono/shape logic so the correct controls are shown for the current mode
        try:
            self.on_mono_changed(self.mono_selector.currentIndex())
        except Exception:
            pass

        # Reset process points and thresholds/tails so plotting starts fresh
        try:
            self.process.points = []
            self.process.lower_threshold = None
            self.process.upper_threshold = None
            self.process.left_tail_value = None
            self.process.right_tail_value = None
        except Exception:
            pass

    def prev_attribute(self):
        """Move to the previous attribute."""
        # ensure any pending debounced apply runs so points reflect UI
        try:
            if hasattr(self, '_apply_timer') and self._apply_timer.isActive():
                self._apply_timer.stop()
            # apply thresholds synchronously to update process.points
            self.apply_thresholds()
        except Exception:
            pass

        # save current attribute state then move; block navigation on validation/save errors
        try:
            meta = self.collect_meta_from_ui()
        except ValueError as e:
            # validation failed; show message and do not navigate
            try:
                self.file_label.setText(str(e))
            except Exception:
                pass
            return
        except Exception as e:
            try:
                self.file_label.setText(f"Error collecting settings: {e}")
            except Exception:
                pass
            return
        try:
            self.process.save_current_state(degree=self.degree_slider.value(), meta=meta)
        except Exception as e:
            try:
                self.file_label.setText(f"Save error: {e}")
            except Exception:
                pass
            return
        if self.process.prev_attribute():
            # reset UI fields so old values are not retained
            self.reset_ui_fields()
            # after moving, load saved state if present, otherwise initialize defaults
            saved = self.process.get_saved_state_for_current_attribute()
            if saved:
                self.apply_saved_state_to_ui(saved)
            else:
                # initialize default points based on (now-empty) inputs and attribute defaults
                try:
                    self.apply_thresholds()
                except Exception:
                    pass
            # refresh the UI (labels/placeholders) and plot
            self.update_ui()

    def next_attribute(self):
        """Move to the next attribute."""
        # ensure any pending debounced apply runs so points reflect UI
        try:
            if hasattr(self, '_apply_timer') and self._apply_timer.isActive():
                self._apply_timer.stop()
            self.apply_thresholds()
        except Exception:
            pass

        # save current attribute state then move; block navigation on validation/save errors
        try:
            meta = self.collect_meta_from_ui()
        except ValueError as e:
            try:
                self.file_label.setText(str(e))
            except Exception:
                pass
            return
        except Exception as e:
            try:
                self.file_label.setText(f"Error collecting settings: {e}")
            except Exception:
                pass
            return
        try:
            self.process.save_current_state(degree=self.degree_slider.value(), meta=meta)
        except Exception as e:
            try:
                self.file_label.setText(f"Save error: {e}")
            except Exception:
                pass
            return
        if self.process.next_attribute():
            # reset UI fields so old values are not retained
            self.reset_ui_fields()
            # after moving, load saved state if present, otherwise initialize defaults
            saved = self.process.get_saved_state_for_current_attribute()
            if saved:
                self.apply_saved_state_to_ui(saved)
            else:
                try:
                    self.apply_thresholds()
                except Exception:
                    pass
            # refresh the UI (labels/placeholders) and plot
            self.update_ui()

    def update_degree_label(self):
        """Update the degree label when the slider changes."""
        self.degree_label.setText(f"Polynomial Degree: {self.degree_slider.value()}")
        self.update_plot()

    def set_param_range(self, iname, minimum, maximum, default=None):
        """Update an existing param control's numeric range and slider position.

        iname: internal param name (as in self.param_controls)
        minimum/maximum: numeric bounds for the logical value
        default: optional default value to set the slider to (falls back to midpoint)
        """
        if iname not in self.param_controls:
            return
        lbl, slider, val_lbl, mn, mx, sc = self.param_controls[iname]
        try:
            mn = float(minimum); mx = float(maximum)
            rng = mx - mn
            steps = float(self._SLIDER_STEPS) if hasattr(self, '_SLIDER_STEPS') else 1000.0
            sc = (rng / steps) if steps > 0 and rng != 0 else 1.0
            self.param_controls[iname] = (lbl, slider, val_lbl, float(mn), float(mx), float(sc))
            # compute slider position for default
            if default is None:
                default = (mn + mx) / 2.0
            pos = int((float(default) - float(mn)) / float(sc)) if float(sc) != 0 else 0
            pos = max(0, min(slider.maximum(), pos))
            slider.blockSignals(True)
            slider.setValue(pos)
            val_lbl.setText(f"{float(default):.4g}")
        finally:
            try:
                slider.blockSignals(False)
            except Exception:
                pass
