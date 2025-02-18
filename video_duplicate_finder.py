#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QListWidget, QCheckBox, QSpinBox,
                            QStyle, QProgressBar, QDialog, QTreeWidget, 
                            QTreeWidgetItem, QMenu, QAction, QMessageBox,
                            QInputDialog, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

class VideoInfo:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.size_mb = self.size / (1024 * 1024)
        self._duration = None
        self._fps = None
        self._resolution = None

    @property
    def duration(self):
        if self._duration is None:
            self._read_video_info()
        return self._duration

    @property
    def fps(self):
        if self._fps is None:
            self._read_video_info()
        return self._fps

    @property
    def resolution(self):
        if self._resolution is None:
            self._read_video_info()
        return self._resolution

    def _read_video_info(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            self._duration = 0
            self._fps = 0
            self._resolution = (0, 0)
            return

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = total_frames / self._fps if self._fps > 0 else 0
        self._resolution = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()

class DuplicateGroup:
    def __init__(self, videos, similarity):
        self.videos = videos  # List of VideoInfo objects
        self.similarity = similarity
        self.selected_for_action = None  # Index of video selected for keeping

    @property
    def total_size(self):
        return sum(v.size for v in self.videos)

    def __len__(self):
        return len(self.videos)

class DragDropList(QListWidget):
    paths_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.accept()
            paths = []
            for url in event.mimeData().urls():
                paths.append(url.toLocalFile())
            self.addPaths(paths)
            self.paths_changed.emit()
        else:
            event.ignore()

    def addPaths(self, paths):
        added = False
        for path in paths:
            if not self.findItems(path, Qt.MatchExactly):
                self.addItem(path)
                added = True
        if added:
            self.paths_changed.emit()

class VideoHasher:
    def __init__(self, video_path, num_frames=10):
        self.video_path = str(video_path)
        self.num_frames = num_frames
        self._info = None
        self._signature = None
        
    @property
    def info(self):
        if self._info is None:
            self._info = VideoInfo(self.video_path)
        return self._info
    
    @property
    def signature(self):
        if self._signature is None:
            self._signature = self.get_video_signature()
        return self._signature
    
    def compute_frame_hash(self, frame):
        small_frame = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        return ''.join(['1' if p > avg else '0' for p in gray.flatten()])
    
    def get_video_signature(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        frame_hashes = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_hashes.append(self.compute_frame_hash(frame))
                
        cap.release()
        return frame_hashes

class DuplicateFinder(QThread):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    progress_max = pyqtSignal(int)
    log_message = pyqtSignal(str)
    stage_progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(list)  # List of DuplicateGroup objects
    
    def __init__(self, paths, recursive=True, similarity_threshold=0.9, num_frames=10):
        super().__init__()
        self.paths = paths
        self.recursive = recursive
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'}
        self.similarity_threshold = similarity_threshold
        self.num_frames = num_frames
    
    def get_video_files(self, path):
        video_files = []
        path = Path(path)
        
        if path.is_file() and path.suffix.lower() in self.video_extensions:
            self.stage_progress.emit("Finding Videos", 1, 1)
            video_files.append(path)
        elif path.is_dir():
            items = list(path.rglob('*') if self.recursive else path.glob('*'))
            for i, filepath in enumerate(items, 1):
                self.stage_progress.emit("Finding Videos", i, len(items))
                if filepath.is_file() and filepath.suffix.lower() in self.video_extensions:
                    video_files.append(filepath)
                    
        return video_files
    
    def compute_similarity(self, hashes1, hashes2):
        if not hashes1 or not hashes2 or len(hashes1) != len(hashes2):
            return 0.0
            
        similarities = []
        for h1, h2 in zip(hashes1, hashes2):
            if len(h1) != len(h2):
                continue
            matches = sum(1 for a, b in zip(h1, h2) if a == b)
            similarity = matches / len(h1)
            similarities.append(similarity)
            
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def get_video_groups(self, videos):
        """Group videos primarily by duration with flexible size matching."""
        duration_groups = {}
        hashers = {}
        
        self.progress.emit("Analyzing video durations...")
        total_videos = len(videos)
        for i, video in enumerate(videos):
            self.stage_progress.emit("Analyzing Videos", i + 1, total_videos)
            
            self.log_message.emit(f"Analyzing [{i+1}/{total_videos}] {video}")
            
            hasher = VideoHasher(video, self.num_frames)
            if not hasher.info.duration:
                self.log_message.emit(f"⚠️ Could not read video information")
                continue
                
            duration = hasher.info.duration
            fps = hasher.info.fps
            
            # Round duration to nearest 0.1 second for more precise grouping
            duration_key = round(duration * 10) / 10
            self.log_message.emit(f"Duration: {duration:.1f}s, FPS: {fps:.1f}")
            
            # Only group by duration, with very tight tolerance (0.5 seconds)
            for d in [duration_key - 0.5, duration_key, duration_key + 0.5]:
                d = round(d * 10) / 10  # Keep 0.1s precision
                if d in duration_groups:
                    duration_groups[d].append(video)
                else:
                    duration_groups[d] = [video]
            
            hashers[video] = hasher
        
        # Log grouping statistics
        total_groups = len(duration_groups)
        avg_group_size = sum(len(g) for g in duration_groups.values()) / total_groups if total_groups > 0 else 0
        large_groups = sum(1 for g in duration_groups.values() if len(g) > 10)
        
        self.log_message.emit(f"\nFormed {total_groups} duration-based groups")
        self.log_message.emit(f"Average group size: {avg_group_size:.1f} videos")
        self.log_message.emit(f"Groups with >10 videos: {large_groups}")
        
        return duration_groups, hashers
    
    def run(self):
        # Collect all video files
        all_videos = []
        scan_count = 0
        for path in self.paths:
            videos = self.get_video_files(path)
            scan_count += 1
            self.progress.emit(f"Scanning path {scan_count}/{len(self.paths)}")
            all_videos.extend(videos)
            self.log_message.emit(f"Found {len(videos)} videos in {path}")
        
        all_videos = list(set(all_videos))  # Remove duplicates
        
        if not all_videos:
            self.log_message.emit("No videos found!")
            self.finished.emit([])
            return
            
        total_size_mb = sum(Path(v).stat().st_size for v in all_videos) / (1024 * 1024)
        self.log_message.emit(f"\nTotal unique videos found: {len(all_videos)}")
        self.log_message.emit(f"Total size: {total_size_mb:.1f}MB")
        
        # Pre-filter videos by duration
        duration_groups, hashers = self.get_video_groups(all_videos)
        
        # Count potential comparisons after grouping
        total_comparisons = 0
        for videos in duration_groups.values():
            n = len(videos)
            total_comparisons += (n * (n - 1)) // 2
            
        self.log_message.emit(f"\nPotential comparisons after filtering: {total_comparisons}")
        self.progress_max.emit(total_comparisons)
        
        # Track which videos have been assigned to groups
        videos_in_groups = set()
        duplicate_groups = []
        comparison_count = 0
        
        # Compare videos within duration groups
        for duration, videos in duration_groups.items():
            for i, video1 in enumerate(videos):
                if video1 in videos_in_groups:
                    continue
                    
                current_group_videos = []
                current_group_similarities = []
                
                for j in range(i + 1, len(videos)):
                    video2 = videos[j]
                    if video2 in videos_in_groups:
                        continue
                        
                    comparison_count += 1
                    if comparison_count % 10 == 0:
                        self.progress.emit(f"Comparing videos {comparison_count}/{total_comparisons}")
                        self.progress_value.emit(comparison_count)
                    
                    # Use cached hashers
                    hasher1 = hashers[video1]
                    hasher2 = hashers[video2]
                    
                    # Get cached signatures
                    sig1 = hasher1.signature
                    sig2 = hasher2.signature
                    
                    if sig1 and sig2:
                        similarity = self.compute_similarity(sig1, sig2)
                        
                        if similarity >= self.similarity_threshold:
                            current_group_videos.append(video2)
                            current_group_similarities.append(similarity)
                            self.log_message.emit(
                                f"\nMatch found between:\n{video1}\n{video2}\n"
                                f"Similarity: {similarity:.2%}")
                
                # If we found any duplicates, create a group
                if current_group_videos:
                    # Add the first video to the group
                    current_group_videos.insert(0, video1)
                    current_group_similarities.insert(0, 1.0)
                    
                    # Mark all videos in this group as processed
                    videos_in_groups.update(current_group_videos)
                    
                    # Create group with VideoInfo objects
                    videos_info = [VideoInfo(v) for v in current_group_videos]
                    avg_similarity = sum(current_group_similarities) / len(current_group_similarities)
                    duplicate_groups.append(DuplicateGroup(videos_info, avg_similarity))
        
        self.progress.emit("Finished scanning!")
        self.log_message.emit(f"\nFound {len(duplicate_groups)} duplicate groups")
        for i, group in enumerate(duplicate_groups):
            self.log_message.emit(f"\nGroup {i+1} ({group.similarity:.1%} similar):")
            for video in group.videos:
                self.log_message.emit(f"- {video.path}")
        
        self.finished.emit(duplicate_groups)

class DuplicateManager(QDialog):
    def __init__(self, duplicate_groups, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duplicate Manager")
        self.setMinimumSize(1000, 600)
        self.duplicate_groups = duplicate_groups
        self.file_actions = {}  # path -> action ("keep", "delete", "skip")
        self.action_log = []
        
        layout = QVBoxLayout(self)
        
        # Action selection
        action_layout = QHBoxLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems([
            "Keep both files",
            "Keep largest resolution",
            "Keep smallest file size",
            "Keep newest file",
            "Keep oldest file",
            "Manual selection"
        ])
        action_layout.addWidget(QLabel("Default Action:"))
        action_layout.addWidget(self.action_combo)
        
        # Apply to all button
        self.apply_all_btn = QPushButton("Apply to All Groups")
        self.apply_all_btn.clicked.connect(self.apply_to_all)
        action_layout.addWidget(self.apply_all_btn)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        # Click instructions
        layout.addWidget(QLabel("Click on individual files to cycle through: Skip → Keep → Delete"))
        
        # Tree widget for duplicates
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Size", "Resolution", "Duration", "Modified Date", "Action"])
        self.tree.setColumnWidth(0, 500)
        self.tree.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.tree)
        
        # Populate tree
        self.populate_tree()
        
        # Add log viewer
        layout.addWidget(QLabel("Action Log:"))
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(100)
        layout.addWidget(self.log_viewer)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Execute Actions")
        save_btn.clicked.connect(self.execute_actions)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def apply_to_all(self):
        action = self.action_combo.currentText()
        self.log_action(f"Applying '{action}' to all groups")
        
        for i, group in enumerate(self.duplicate_groups):
            if action == "Keep both files":
                # Set all files in group to Keep
                for video in group.videos:
                    path = str(video.path)
                    self.file_actions[path] = "Keep"
            else:
                selected_idx = self.get_selected_index(group, action)
                # Set selected file to Keep, others to Delete
                for j, video in enumerate(group.videos):
                    path = str(video.path)
                    self.file_actions[path] = "Keep" if j == selected_idx else "Delete"
        
        self.update_tree()
    
    def get_selected_index(self, group, action):
        """Determine which file to keep based on the selected action."""
        if action == "Keep largest resolution":
            return max(range(len(group.videos)), 
                      key=lambda i: group.videos[i].resolution[0] * group.videos[i].resolution[1])
        elif action == "Keep smallest file size":
            return min(range(len(group.videos)), 
                      key=lambda i: group.videos[i].size)
        elif action == "Keep newest file":
            return max(range(len(group.videos)), 
                      key=lambda i: group.videos[i].path.stat().st_mtime)
        elif action == "Keep oldest file":
            return min(range(len(group.videos)), 
                      key=lambda i: group.videos[i].path.stat().st_mtime)
        return 0  # Default to first file for manual selection

    def log_action(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.action_log.append(log_entry)
        self.log_viewer.append(log_entry)
    
    def format_duration(self, seconds):
        return f"{int(seconds/60)}:{int(seconds%60):02d}"
    
    def populate_tree(self):
        for i, group in enumerate(self.duplicate_groups):
            group_item = QTreeWidgetItem(self.tree)
            group_item.setText(0, f"Group {i+1} - {len(group)}x duplicates ({group.similarity:.1%} similar)")
            group_item.setExpanded(True)
            
            for j, video in enumerate(group.videos):
                video_item = QTreeWidgetItem(group_item)
                video_item.setText(0, str(video.path))
                video_item.setText(1, f"{video.size_mb:.1f} MB")
                video_item.setText(2, f"{video.resolution[0]}x{video.resolution[1]}")
                video_item.setText(3, self.format_duration(video.duration))
                mtime = datetime.fromtimestamp(video.path.stat().st_mtime)
                video_item.setText(4, mtime.strftime("%Y-%m-%d %H:%M:%S"))
                video_item.setText(5, "Keep")  # Default action
                video_item.setForeground(5, Qt.green)
                video_item.setData(0, Qt.UserRole, str(video.path))
    
    def cycle_action(self, item):
        """Toggle between Keep and Delete"""
        current_action = item.text(5)
        path = item.data(0, Qt.UserRole)
        
        if current_action == "Delete":
            new_action = "Keep"
            color = Qt.green
        else:  # Keep or empty
            new_action = "Delete"
            color = Qt.red
            
        item.setText(5, new_action)
        item.setForeground(5, color)
        self.file_actions[path] = new_action
        self.log_action(f"Set action for {Path(path).name}: {new_action}")
    
    def on_item_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if path:  # If this is a video item
            self.cycle_action(item)
    
    def update_tree(self):
        """Update the tree to reflect current file actions"""
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                path = child.data(0, Qt.UserRole)
                if path in self.file_actions:
                    action = self.file_actions[path]
                    child.setText(5, action)
                    if action == "Keep":
                        child.setForeground(5, Qt.green)
                    elif action == "Delete":
                        child.setForeground(5, Qt.red)
                    else:  # Skip
                        child.setForeground(5, Qt.gray)
    
    def execute_actions(self):
        actions_to_execute = {path: action for path, action in self.file_actions.items() 
                            if action == "Delete"}  # Only process Delete actions
        
        if not actions_to_execute:
            response = QMessageBox.question(
                self, "No Actions", 
                "No files selected for deletion. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.No:
                return
        
        # Create backup folder
        backup_dir = Path("video_duplicates_backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Save action log
        log_file = backup_dir / "action_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Video Duplicate Manager - Action Log\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(self.action_log))
            f.write("\n\nExecuted Actions:\n")
            f.write("=" * 50 + "\n\n")
        
        # Execute actions
        total_saved = 0
        files_moved = 0
        errors = []
        
        with open(log_file, 'a', encoding='utf-8') as f:
            # Group files by their groups for better logging
            for i, group in enumerate(self.duplicate_groups):
                f.write(f"\nGroup {i+1}:\n")
                group_files = [str(v.path) for v in group.videos]
                
                for path_str in group_files:
                    action = self.file_actions.get(path_str, "Keep")
                    path = Path(path_str)
                    
                    if action == "Delete":
                        try:
                            # Get file size before moving
                            file_size = path.stat().st_size
                            
                            backup_path = backup_dir / path.name
                            if backup_path.exists():
                                backup_path = backup_dir / f"{path.stem}_duplicate{path.suffix}"
                                
                            try:
                                shutil.move(str(path), str(backup_path))
                                total_saved += file_size
                                files_moved += 1
                                f.write(f"Moved to backup: {path}\n")
                            except Exception as e:
                                errors.append(f"Error moving {path}: {str(e)}")
                                f.write(f"ERROR moving {path}: {str(e)}\n")
                        except Exception as e:
                            errors.append(f"Error processing {path}: {str(e)}")
                            f.write(f"ERROR processing {path}: {str(e)}\n")
                    elif action == "Keep":
                        f.write(f"Kept: {path}\n")
        
        # Show results with any errors
        result_message = (
            f"Actions executed!\n"
            f"Files moved to backup: {files_moved}\n"
            f"Space saved: {total_saved / (1024*1024*1024):.1f} GB\n"
            f"Files backed up to: {backup_dir}\n"
            f"Log file: {log_file}"
        )
        
        if errors:
            result_message += "\n\nErrors occurred:\n" + "\n".join(errors)
            QMessageBox.warning(self, "Completed with Errors", result_message)
        else:
            QMessageBox.information(self, "Success", result_message)
            
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Duplicate Finder")
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Path list
        path_layout = QVBoxLayout()
        path_header = QHBoxLayout()
        path_header.addWidget(QLabel("Drag and drop files/folders here:"))
        add_button = QPushButton("Add Files/Folders")
        add_button.clicked.connect(self.add_paths)
        path_header.addWidget(add_button)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_paths)
        path_header.addWidget(clear_button)
        path_layout.addLayout(path_header)
        
        self.path_list = DragDropList()
        path_layout.addWidget(self.path_list)
        
        # Settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Similarity Threshold (%):"))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(50, 100)
        self.threshold_spinbox.setValue(90)
        settings_layout.addWidget(self.threshold_spinbox)
        
        self.recursive_checkbox = QCheckBox("Search Recursively")
        self.recursive_checkbox.setChecked(True)
        settings_layout.addWidget(self.recursive_checkbox)
        settings_layout.addStretch()
        
        # Start button
        self.start_button = QPushButton("Start Scanning")
        self.start_button.clicked.connect(self.start_scan)
        self.start_button.setEnabled(False)
        
        # Stage progress
        stage_layout = QHBoxLayout()
        self.stage_label = QLabel("Stage: ")
        self.stage_progress = QProgressBar()
        stage_layout.addWidget(self.stage_label)
        stage_layout.addWidget(self.stage_progress)
        
        # Overall progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(200)
        
        # Results area
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        
        # Add widgets to layout
        layout.addLayout(path_layout)
        layout.addLayout(settings_layout)
        layout.addWidget(self.start_button)
        layout.addLayout(stage_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Detailed Log:"))
        layout.addWidget(self.log_area)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_area)
        
        # Connect path_list signals
        self.path_list.itemSelectionChanged.connect(self.update_start_button)
        self.path_list.paths_changed.connect(self.update_start_button)
    
    def add_paths(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        if paths:
            self.path_list.addPaths(paths)
    
    def clear_paths(self):
        self.path_list.clear()
        self.update_start_button()
    
    def update_start_button(self):
        self.start_button.setEnabled(self.path_list.count() > 0)
    
    def update_progress(self, message):
        self.progress_bar.setFormat(f"{message} - %p%")
    
    def set_progress_max(self, maximum):
        self.progress_bar.setMaximum(maximum)
    
    def set_progress_value(self, value):
        self.progress_bar.setValue(value)
    
    def update_stage_progress(self, stage_name, current, total):
        self.stage_label.setText(f"Stage: {stage_name}")
        self.stage_progress.setMaximum(total)
        self.stage_progress.setValue(current)
        
    def log_detail(self, message):
        self.log_area.append(message)
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_results(self, duplicate_groups):
        if not duplicate_groups:
            self.results_area.setText("No duplicate videos found.")
            QMessageBox.information(self, "Scan Complete", "No duplicate videos found.")
        else:
            total_groups = len(duplicate_groups)
            total_duplicates = sum(len(group.videos) - 1 for group in duplicate_groups)
            potential_savings = sum(group.total_size - min(v.size for v in group.videos) 
                                 for group in duplicate_groups)
            
            self.results_area.setText(
                f"Found {total_groups} groups of duplicates\n"
                f"Total duplicate files: {total_duplicates}\n"
                f"Potential space savings: {potential_savings / (1024*1024*1024):.1f} GB\n\n"
            )
            
            # Show duplicate manager
            manager = DuplicateManager(duplicate_groups, self)
            manager.exec_()
        
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.stage_progress.setValue(0)
        self.stage_label.setText("Stage: Complete")

    def start_scan(self):
        paths = [self.path_list.item(i).text() for i in range(self.path_list.count())]
        if not paths:
            return
        
        self.start_button.setEnabled(False)
        self.log_area.clear()
        self.results_area.clear()
        self.stage_progress.setValue(0)
        self.progress_bar.setValue(0)
        
        similarity_threshold = self.threshold_spinbox.value() / 100.0
        self.finder_thread = DuplicateFinder(
            paths, 
            self.recursive_checkbox.isChecked(),
            similarity_threshold
        )
        self.finder_thread.progress.connect(self.update_progress)
        self.finder_thread.progress_max.connect(self.set_progress_max)
        self.finder_thread.progress_value.connect(self.set_progress_value)
        self.finder_thread.log_message.connect(self.log_detail)
        self.finder_thread.stage_progress.connect(self.update_stage_progress)
        self.finder_thread.finished.connect(self.show_results)
        self.finder_thread.start()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
