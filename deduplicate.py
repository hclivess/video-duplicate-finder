import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QListWidget, QCheckBox, QSpinBox,
                            QStyle, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

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
        self._video_info = None
        self._signature = None
        
    @property
    def video_info(self):
        if self._video_info is None:
            self._video_info = self.get_video_info()
        return self._video_info
        
    @property
    def signature(self):
        if self._signature is None:
            self._signature = self.get_video_signature()
        return self._signature
    
    def get_video_info(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }
    
    def compute_frame_hash(self, frame):
        small_frame = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        return ''.join(['1' if p > avg else '0' for p in gray.flatten()])
    
    def get_video_signature(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
            
        info = self.get_video_info()
        if not info or info['total_frames'] <= 0:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, info['total_frames']-1, self.num_frames, dtype=int)
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
    stage_progress = pyqtSignal(str, int, int)  # stage name, current, total
    finished = pyqtSignal(list)
    
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
            
            try:
                size_mb = Path(video).stat().st_size / (1024 * 1024)
            except Exception:
                self.log_message.emit(f"⚠️ Could not read file size for {video.name}")
                continue
            
            self.log_message.emit(f"Analyzing [{i+1}/{total_videos}] {video.name} ({size_mb:.1f}MB)")
            
            hasher = VideoHasher(video, self.num_frames)
            info = hasher.video_info
            if not info:
                self.log_message.emit(f"⚠️ Could not read video information")
                continue
                
            duration = info['duration']
            fps = info['fps']
            
            # Round duration to nearest 0.1 second for more precise grouping
            duration_key = round(duration * 10) / 10
            self.log_message.emit(f"Duration: {duration:.1f}s, FPS: {fps:.1f}")
            
            # Only group by duration, with very tight tolerance (±0.5 seconds)
            for d in [duration_key - 0.5, duration_key, duration_key + 0.5]:
                d = round(d * 10) / 10  # Keep 0.1s precision
                if d in duration_groups:
                    # Log size difference if it's notable but still include in group
                    existing_sizes = [Path(v).stat().st_size / (1024 * 1024) for v in duration_groups[d]]
                    avg_size = sum(existing_sizes) / len(existing_sizes)
                    if abs(size_mb - avg_size) > 50:  # If size differs by more than 50MB
                        self.log_message.emit(f"Note: Large size difference in group ({size_mb:.1f}MB vs avg {avg_size:.1f}MB)")
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
        
        # Log largest groups for inspection
        sorted_groups = sorted(duration_groups.items(), key=lambda x: len(x[1]), reverse=True)
        if sorted_groups:
            self.log_message.emit("\nLargest groups by duration:")
            for duration, group in sorted_groups[:5]:
                self.log_message.emit(f"{duration:.1f}s: {len(group)} videos")
        
        return duration_groups, hashers
        
        return groups, hashers

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
        total_size_mb = sum(Path(v).stat().st_size for v in all_videos) / (1024 * 1024)
        self.log_message.emit(f"\nTotal unique videos found: {len(all_videos)}")
        self.log_message.emit(f"Total size: {total_size_mb:.1f}MB")
        
        # Pre-filter videos by duration and size
        groups, hashers = self.get_video_groups(all_videos)
        
        # Count potential comparisons after grouping
        total_comparisons = 0
        for videos in groups.values():
            n = len(videos)
            total_comparisons += (n * (n - 1)) // 2
            
        self.log_message.emit(f"\nPotential comparisons after filtering: {total_comparisons}")
        self.progress_max.emit(total_comparisons)
        
        # Compare videos within groups
        duplicates = []
        comparison_count = 0
        
        # Compare only within same groups
        for group_key, videos in groups.items():
            duration, size = group_key
            for i, video1 in enumerate(videos):
                for j in range(i + 1, len(videos)):
                    video2 = videos[j]
                    comparison_count += 1
                    
                    if comparison_count % 100 == 0:  # Update progress less frequently
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
                            self.log_message.emit(f"\nMatch found between:\n{video1}\n{video2}\nSimilarity: {similarity:.2%}")
                            duplicates.append((str(video1), str(video2), similarity))
    
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
            
        self.log_message.emit(f"\nPotential comparisons after duration filtering: {total_comparisons}")
        self.progress_max.emit(total_comparisons)
        
        # Compare videos within duration groups
        duplicates = []
        comparison_count = 0
        
        # Compare only within duration groups
        for duration, videos in duration_groups.items():
            for i, video1 in enumerate(videos):
                for j in range(i + 1, len(videos)):
                    video2 = videos[j]
                    comparison_count += 1
                    self.progress.emit(f"Comparing videos {comparison_count}/{total_comparisons}")
                    self.progress_value.emit(comparison_count)
                    
                    # Use cached hashers
                    hasher1 = hashers[video1]
                    hasher2 = hashers[video2]
                    
                    # Quick check of video info before full comparison
                    info1 = hasher1.video_info
                    info2 = hasher2.video_info
                    
                    if not info1 or not info2:
                        self.log_message.emit("Error reading one or both videos")
                        continue
                        
                    # Get cached signatures
                    sig1 = hasher1.signature
                    sig2 = hasher2.signature
                    
                    if sig1 and sig2:
                        similarity = self.compute_similarity(sig1, sig2)
                        
                        if similarity >= self.similarity_threshold:
                            self.log_message.emit(f"\nMatch found between:\n{video1}\n{video2}\nSimilarity: {similarity:.2%}")
                            duplicates.append((str(video1), str(video2), similarity))
                    else:
                        self.log_message.emit("Error processing video signatures")
        
        self.progress.emit("Finished scanning!")
        self.finished.emit(duplicates)

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

    def show_results(self, duplicates):
        if not duplicates:
            self.results_area.setText("No duplicate videos found.")
        else:
            result_text = "Duplicate videos found:\n\n"
            duplicates.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
            for file1, file2, similarity in duplicates:
                # Get file sizes
                size1 = Path(file1).stat().st_size / (1024 * 1024)  # MB
                size2 = Path(file2).stat().st_size / (1024 * 1024)  # MB
                result_text += f"Match ({similarity:.2%} similar):\n"
                result_text += f"1. {file1} ({size1:.1f}MB)\n"
                result_text += f"2. {file2} ({size2:.1f}MB)\n\n"
            self.results_area.setText(result_text)
        
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
