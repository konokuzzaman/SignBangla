"""Script to generate a publication-quality presentation progress PDF
following the official North South University (NSU) ECE Department
Junior Design Project (CSE299) Weekly Update format.

Exact 7-Page Presentation Structure:
Page 1: Official NSU Title Cover Page (ECE Dept, NSU Logo, Project Title in Red, Members, Advisor Dr. Nabil Bin Hannan)
Page 2: Weekly Update Executive Summary, Meeting Notes, Architectural Flow & Master 10-Week Status Dashboard
Page 3: Member 1 (MD. Konok-Uz-Zaman: Data & Model Lead) - Ingestion, Keypoint Extraction, Slicing & Quality QA
Page 4: Member 2 (Wahiduzzaman Mahin: Training & Evaluation Lead) - Normalization, DataLoaders, Augmentation & Comparative Training
Page 5: Member 3 (Shesir Baidya: Real-Time System & Demo Lead) - Live Webcam System, HUD & Module 2 Speech-to-Sign Engine
Page 6: Experimental Benchmark Results & In-Depth Diagnostic Analysis (BiGRU vs BiLSTM, Confusion Matrix, Worst Classes)
Page 7: System Integration Verification Checklist, CLI Execution Commands, Roadmap & Faculty Advisor Sign-Off
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# ==========================================
# METADATA & CONFIGURATION
# ==========================================
PROJECT_TITLE = "SignBangla: Real-Time Bangla Sign Language Recognition and Speech System"
UPDATE_TITLE = "CSE299 Project Weekly Update [#2]"
UPDATE_PERIOD = "Weeks 1–5 Comprehensive Benchmark & Dual-Module Update"
TERM_STR = "Fall, 2026"

MEMBERS = [
    {"name": "MD. Konok-Uz-Zaman", "id": "ID # [xxxxxxxxxx]", "role": "Member 1", "title": "Data & Model Lead"},
    {"name": "Wahiduzzaman Mahin", "id": "ID # [xxxxxxxxxx]", "role": "Member 2", "title": "Training & Evaluation Lead"},
    {"name": "Shesir Baidya", "id": "ID # [xxxxxxxxxx]", "role": "Member 3", "title": "Real-Time System & Demo Lead"},
]

ADVISOR_NAME = "Dr. Nabil Bin Hannan"
ADVISOR_TITLE = "Assistant Professor"
ADVISOR_DEPT = "ECE Department"

# Color Palette (Matching NSU Branding & Professional Dark Palette)
NAVY = colors.HexColor("#0B2545")
CRIMSON = colors.HexColor("#C00000")
TEAL = colors.HexColor("#0D9488")
SLATE = colors.HexColor("#334155")
CARD_BG = colors.HexColor("#F8FAFC")
BORDER_COLOR = colors.HexColor("#CBD5E1")
ACCENT_GREEN = colors.HexColor("#15803D")
WHITE = colors.HexColor("#FFFFFF")
GRAY_TEXT = colors.HexColor("#64748B")
AMBER_BG = colors.HexColor("#FEF3C7")
GREEN_BG = colors.HexColor("#F0FDF4")


class NSUNumberedCanvas(canvas.Canvas):
    """Two-pass canvas providing NSU standard running headers and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(GRAY_TEXT)
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(40, 762, "CSE299 Project Weekly Update [#2] — SignBangla")
            self.drawRightString(572, 762, f"{TERM_STR} | {UPDATE_PERIOD}")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(40, 756, 572, 756)
        
        # Running Footer (All Pages)
        if self._pageNumber > 1:
            page_str = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(572, 20, page_str)
            self.drawString(40, 20, "Department of Electrical and Computer Engineering — North South University")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(40, 30, 572, 30)
        
        self.restoreState()


def create_presentation_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=38,
        rightMargin=38,
        topMargin=34,
        bottomMargin=34
    )

    styles = getSampleStyleSheet()
    
    # Typography Styles
    style_cover_dept = ParagraphStyle(
        "CoverDept",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        alignment=1,
        textColor=NAVY
    )
    style_cover_univ = ParagraphStyle(
        "CoverUniv",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        alignment=1,
        textColor=NAVY
    )
    style_cover_course = ParagraphStyle(
        "CoverCourse",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        alignment=1,
        textColor=SLATE
    )
    style_cover_update = ParagraphStyle(
        "CoverUpdate",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=18,
        alignment=1,
        textColor=SLATE
    )
    style_cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=21,
        alignment=1,
        textColor=CRIMSON
    )
    style_cover_meta = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=SLATE
    )
    style_cover_meta_bold = ParagraphStyle(
        "CoverMetaBold",
        parent=style_cover_meta,
        fontName="Helvetica-Bold"
    )

    # Inner Pages Styles
    style_page_heading = ParagraphStyle(
        "PageHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=16.5,
        textColor=NAVY,
        spaceAfter=2
    )
    style_page_sub = ParagraphStyle(
        "PageSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=TEAL,
        spaceAfter=4
    )
    style_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11.5,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=2,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.4,
        textColor=SLATE,
        spaceAfter=2
    )
    style_body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=style_body,
        fontName="Helvetica-Bold"
    )
    style_caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.8,
        leading=8.5,
        textColor=GRAY_TEXT,
        alignment=1,
        spaceBefore=1,
        spaceAfter=2
    )
    style_card = ParagraphStyle(
        "CardText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=SLATE
    )
    style_code = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.8,
        leading=8.5,
        textColor=NAVY
    )
    style_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8.6,
        textColor=SLATE
    )
    style_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=style_cell,
        fontName="Helvetica-Bold"
    )

    story = []

    # =========================================================================
    # PAGE 1: OFFICIAL NSU COVER PAGE (MATCHING TEMPLATE EXACTLY)
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("Department of Electrical and Computer Engineering", style_cover_dept))
    story.append(Paragraph("North South University", style_cover_univ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceBefore=0, spaceAfter=18))

    # Centered Official NSU Logo Monogram
    p_logo = Path("nsu_logo.png")
    if p_logo.exists():
        img_logo = Image(str(p_logo), width=85, height=102)
        img_logo.hAlign = 'CENTER'
        story.append(img_logo)
    else:
        story.append(Spacer(1, 102))

    story.append(Spacer(1, 20))
    story.append(Paragraph("Junior Design Project", style_cover_course))
    story.append(Paragraph("CSE299 Project Weekly Update [#2]", style_cover_update))
    story.append(Spacer(1, 24))

    # Project Title in Red Bold
    story.append(Paragraph("SignBangla: Real-Time Bangla Sign Language<br/>Recognition and Speech System", style_cover_title))
    story.append(Spacer(1, 32))

    # Member Table (Aligned)
    member_table_data = []
    for m in MEMBERS:
        member_table_data.append([
            Paragraph(f"<b>{m['name']}</b>", style_cover_meta_bold),
            Paragraph(f"<b>{m['id']}</b>", style_cover_meta)
        ])
    t_members = Table(member_table_data, colWidths=[200, 160], hAlign='CENTER')
    t_members.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_members)
    story.append(Spacer(1, 36))

    # Faculty Advisor Block
    story.append(Paragraph("<b>Faculty Advisor:</b>", style_cover_meta_bold))
    story.append(Spacer(1, 2))
    story.append(Paragraph(f"<b>{ADVISOR_NAME}</b>", style_cover_meta_bold))
    story.append(Paragraph(ADVISOR_TITLE, style_cover_meta))
    story.append(Paragraph(ADVISOR_DEPT, style_cover_meta))
    story.append(Spacer(1, 36))

    # Term / Year
    story.append(Paragraph(f"<b>{TERM_STR}</b>", style_cover_meta_bold))

    # =========================================================================
    # PAGE 2: EXECUTIVE SUMMARY, MEETING LOG & MASTER TIMELINE DASHBOARD
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph(f"{UPDATE_PERIOD}", style_page_heading))
    story.append(Paragraph(
        "This week our group met <b>3 times</b> (twice in-person in the ECE Senior Design Lab and once online via Discord) "
        "where we integrated, benchmarked, and evaluated our foundational engineering modules. This presentation reflects our "
        "cumulative progress after <b>5 weeks of active execution</b> following proposal approval. "
        "<b>Phase 1 (data pipeline, Weeks 1–3) and Phase 2 (baseline neural sequence models) are now 100% complete across all 60 words and 18 signers</b>. "
        "Both <b>Module 1</b> (Sign &rarr; Speech via deep BiGRU and BiLSTM models achieving up to <b>86.79% test accuracy</b> on held-out signers) "
        "and <b>Module 2</b> (Speech &rarr; Sign via dictionary compilation, kinematic motion stitching, and H.264 video rendering with native Bangla typography) "
        "are fully operational, verified, and demonstrated. Here is the detailed engineering record for this update —",
        style_body
    ))
    story.append(Spacer(1, 2))

    # Team Overview Table
    team_data = [
        [
            Paragraph("<b>Member</b>", style_body_bold),
            Paragraph("<b>Name & Title</b>", style_body_bold),
            Paragraph("<b>Primary Technical Responsibilities (Weeks 1-5)</b>", style_body_bold),
            Paragraph("<b>Status</b>", style_body_bold)
        ]
    ]
    for m in MEMBERS:
        team_data.append([
            Paragraph(f"<b>{m['role']}</b>", style_body),
            Paragraph(f"<b>{m['name']}</b><br/><font color='#64748B' size=6.5>{m['title']}</font>", style_body),
            Paragraph(
                "Dataset Inspection, 60-Word Keypoint Extraction, Trial Slicing, Quality QA & BiGRU/BiLSTM Formulations" if m['role'] == "Member 1" else
                ("Normalization, Resampling, DataLoaders, Augmentations & BiGRU/BiLSTM Training Engine (MPS)" if m['role'] == "Member 2" else
                 "Signer Splits, Live Webcam HUD, and Module 2 Speech/Text-to-Sign Animation Engine"),
                style_body
            ),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font><br/><font size=5.5 color='#64748B'>Weeks 1-5 Verified</font>", style_body)
        ])
    t_team = Table(team_data, colWidths=[65, 145, 230, 96])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 3))

    # Master Timeline Grid (Weeks 1-10)
    story.append(Paragraph("Master 10-Week Engineering Timeline & Milestone Status", style_h2))
    timeline_data = [
        [
            Paragraph("<b>Wk</b>", style_body_bold),
            Paragraph("<b>Milestone Phase</b>", style_body_bold),
            Paragraph("<b>Member 1 Focus</b>", style_body_bold),
            Paragraph("<b>Member 2 Focus</b>", style_body_bold),
            Paragraph("<b>Member 3 Focus</b>", style_body_bold),
            Paragraph("<b>Status</b>", style_body_bold)
        ],
        [
            Paragraph("<b>1</b>", style_body),
            Paragraph("Dataset & Setup", style_body),
            Paragraph("Download & inspect BdSLW60", style_body),
            Paragraph("GitHub & environment setup", style_body),
            Paragraph("Test MediaPipe with webcam", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>2</b>", style_body),
            Paragraph("Keypoint Extraction", style_body),
            Paragraph("Develop extraction script", style_body),
            Paragraph("Develop preprocessing format", style_body),
            Paragraph("Prepare signer-wise split", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>3</b>", style_body),
            Paragraph("Dataset Preparation", style_body),
            Paragraph("Check keypoint quality", style_body),
            Paragraph("DataLoader & augmentation", style_body),
            Paragraph("Evaluation metrics suite", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>4</b>", style_body),
            Paragraph("Baseline Models", style_body),
            Paragraph("BiGRU & BiLSTM architectures", style_body),
            Paragraph("Train & evaluate (86.79% Acc)", style_body),
            Paragraph("Live HUD & Module 2 Demo", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>5</b>", style_body),
            Paragraph("Dual-Module System", style_body),
            Paragraph("Model diagnostics & confusion", style_body),
            Paragraph("BiGRU vs BiLSTM validation", style_body),
            Paragraph("Interactive HUD & stitching", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>6</b>", style_body),
            Paragraph("Main Transformer", style_body),
            Paragraph("Implement ST-Transformer", style_body),
            Paragraph("Hyperparameter tuning", style_body),
            Paragraph("Sign temporal smoothing", style_body),
            Paragraph("<font color='#D97706'><b>CURRENT</b></font>", style_body)
        ],
        [
            Paragraph("<b>7</b>", style_body),
            Paragraph("Speech Integration", style_body),
            Paragraph("Sign dictionary expansion", style_body),
            Paragraph("Bangla ASR model setup", style_body),
            Paragraph("gTTS speech output", style_body),
            Paragraph("<font color='#64748B'>Planned</font>", style_body)
        ],
        [
            Paragraph("<b>8-10</b>", style_body),
            Paragraph("Final Integration", style_body),
            Paragraph("Bi-directional demo pipeline", style_body),
            Paragraph("Comprehensive benchmarking", style_body),
            Paragraph("Final report & presentation", style_body),
            Paragraph("<font color='#64748B'>Planned</font>", style_body)
        ]
    ]
    t_timeline = Table(timeline_data, colWidths=[24, 96, 134, 136, 106, 40])
    t_timeline.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, 5), GREEN_BG),
        ('BACKGROUND', (0, 6), (-1, 6), AMBER_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t_timeline)
    story.append(Spacer(1, 3))

    # Architecture Overview Callout
    arch_box = [
        [
            Paragraph("<b>Dual-Module System Engineering Flow:</b><br/>"
                      "• <b>Module 1 (Sign &rarr; Speech):</b> Video Ingestion &rarr; MediaPipe Holistic (67 3D landmarks: 25 pose + 21 LH + 21 RH) &rarr; "
                      "Shoulder Centering & Scale Invariance &rarr; 64-Frame Resampling &rarr; Continuous Augmentations &rarr; "
                      "BiGRU / BiLSTM with Temporal Attention (<b>86.79% Test Acc</b> on unseen signers U4 & U8) &rarr; 30+ FPS Live Bilingual Webcam HUD.<br/>"
                      "• <b>Module 2 (Speech &rarr; Sign):</b> Speech / Text Input &rarr; Bangla Word Tokenization &rarr; "
                      "60-Word Sign Dictionary Lookup &rarr; Multi-sign Transition Stitching (Cosine / Minimum-Jerk) &rarr; "
                      "H.264 Animated Skeleton Video with Native Bangla Typography.",
                      style_card)
        ]
    ]
    t_arch = Table(arch_box, colWidths=[536])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, TEAL),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_arch)

    # =========================================================================
    # PAGE 3: MEMBER 1 DETAILED ACCOMPLISHMENTS (MD. KONOK-UZ-ZAMAN)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("1. Group Member 1: MD. Konok-Uz-Zaman (Data & Model Lead)", style_page_heading))
    story.append(Paragraph("Role: Dataset Ingestion, 60-Word Keypoint Extraction, Quality Assurance & Neural Architectures", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=4))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 5)", style_h2))
    m1_bullets = (
        "<b>a) 60-Word Dataset Ingestion & Video Inventory:</b> Ingested the complete BdSLW60 benchmark dataset comprising 1,080 continuous "
        "video recordings across 18 signers and 60 isolated vocabulary words. Built atomic tokenization scripts resolving complex filename "
        "stems (<code>U{signer}W{word}F.mp4</code>) and 60 accompanying JSON annotation files.<br/>"
        "<b>b) MediaPipe Holistic 67-Keypoint Extraction:</b> Implemented robust 67-landmark 3D coordinate extraction (25 upper-body pose + "
        "21 left hand + 21 right hand), dropping legs and the 468-point face mesh to prevent appearance bias and ensure real-time 30+ FPS speed. "
        "Identified and rectified <b>18 upside-down videos</b> (e.g. U17W351F–U17W360F, U9W96F, U12W220F) by rotating frames 180&deg; upright.<br/>"
        "<b>c) Repetition Boundary Slicing:</b> Developed an adaptive regex boundary parser to handle varied JSON timestamp keys ('actual start', "
        "'gesture end') across annotators, slicing continuous videos into <b>9,307 raw sign repetition trials</b>.<br/>"
        "<b>d) Quality Auditing & Clean Dataset Assembly:</b> Automated multi-stage quality assurance (<code>scripts/05_check_quality.py</code>). "
        "Flagged and filtered 1,022 defective repetitions with severe occlusion or lost hands; curated <b>8,285 high-integrity trials</b> "
        "(5,558 train, 1,667 val, 1,060 test) with zero missing-body NaN errors.<br/>"
        "<b>e) Neural Sequence Model Formulations:</b> Authored deep sequence classifiers in <code>signbangla/models.py</code>: "
        "<code>BiGRUClassifier</code> (594k params) and <code>BiLSTMClassifier</code> (760k params) featuring spatial input projection, "
        "LayerNorm, 2-layer bidirectional recurrent cores, and learned <b>Temporal Attention pooling</b>."
    )
    story.append(Paragraph(m1_bullets, style_body))
    story.append(Spacer(1, 2))

    story.append(Paragraph("B. Verification Visualizations & Extraction Artifacts", style_h2))
    p_aug = Path("work/quality_plots/augmentation_preview.png")
    if p_aug.exists():
        story.append(Image(str(p_aug), width=510, height=78))
        story.append(Paragraph("Figure 1.1: Geometric data augmentation preview: normalized base skeleton followed by 5 continuous augmented variants (rotation, scale, shift, time warp).", style_caption))

    p_summary = Path("work/quality_plots/summary.png")
    if p_summary.exists():
        story.append(Image(str(p_summary), width=510, height=75))
        story.append(Paragraph("Figure 1.2: Sign trial duration distribution and dominant hand breakdown from automated quality audit.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>scripts/download_dataset.py</code>, <code>scripts/01_inspect_dataset.py</code>, <code>scripts/03_extract_keypoints.py</code>, <code>scripts/04_build_trials.py</code>, <code>scripts/05_check_quality.py</code>, <code>signbangla/models.py</code>", style_card))

    # =========================================================================
    # PAGE 4: MEMBER 2 DETAILED ACCOMPLISHMENTS (WAHIDUZZAMAN MAHIN)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Group Member 2: Wahiduzzaman Mahin (Training & Evaluation Lead)", style_page_heading))
    story.append(Paragraph("Role: Geometric Normalization, Uniform Resampling, PyTorch DataLoaders & Comparative Training", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=4))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 5)", style_h2))
    m2_bullets = (
        "<b>a) Environment Architecture & Testing Harness:</b> Established virtual environment with strictly pinned <code>mediapipe==0.10.14</code> "
        "and PyTorch with Apple Silicon GPU acceleration (<code>mps</code>). Authored a 20-test synthetic self-test suite (<code>tests/test_pipeline.py</code>) "
        "running 100% self-contained verification without dataset dependencies.<br/>"
        "<b>b) Geometric Invariance & Coordinate Transformation:</b> Implemented rigorous coordinate normalization ensuring spatial invariance: "
        "aspect ratio correction, translation centering at shoulder midpoint <code>(left_shoulder + right_shoulder)/2</code>, and distance scaling "
        "by Euclidean shoulder width <code>||left_shoulder - right_shoulder||</code>. Eliminates variance from signer physique and camera distance.<br/>"
        "<b>c) Temporal Gap Interpolation & Uniform 64-Frame Resampling:</b> Resolved tracking drops via linear gap interpolation (&le;6 frames for hands, "
        "&le;15 frames for pose). Resampled all variable-length trials to uniform 64-frame tensors <code>(64, 67, 3)</code>, flattening to 201 features.<br/>"
        "<b>d) PyTorch DataLoader with 6-Fold Augmentations:</b> Developed <code>SignBanglaDataset</code> and <code>make_dataloaders</code>. "
        "Implemented training-only continuous augmentations: rotation (&plusmn;10&deg;), scale (0.9–1.1&times;), shift (&plusmn;0.10), "
        "temporal cropping (up to 10%), time warping (&plusmn;25%), and Gaussian jitter (std 0.01) with zero NaN artifacts.<br/>"
        "<b>e) Dual-Model Training Engine & Benchmarking:</b> Engineered <code>scripts/10_train_baseline.py</code> supporting AdamW, "
        "CosineAnnealingLR, and dual architectures. Successfully trained both <b>BiGRU (86.79% Test Acc)</b> and <b>BiLSTM (86.04% Test Acc)</b> "
        "on the full 60-word split on Apple Silicon GPU in under 12 minutes."
    )
    story.append(Paragraph(m2_bullets, style_body))
    story.append(Spacer(1, 2))

    story.append(Paragraph("B. Coordinate Normalization & Resampling Specifications", style_h2))
    prep_specs = [
        [Paragraph("<b>Normalization Step</b>", style_body_bold), Paragraph("<b>Mathematical Formulation</b>", style_body_bold), Paragraph("<b>Engineering Objective</b>", style_body_bold)],
        [
            Paragraph("Aspect Ratio Correction", style_body),
            Paragraph("x' = x * (W/H),&nbsp;&nbsp;z' = z * (W/H)", style_body),
            Paragraph("Equalizes spatial measurement units along x, y, and z axes.", style_body)
        ],
        [
            Paragraph("Origin Translation", style_body),
            Paragraph("origin = (left_shoulder + right_shoulder) / 2", style_body),
            Paragraph("Invariance against subject position within camera frame.", style_body)
        ],
        [
            Paragraph("Scale Invariance", style_body),
            Paragraph("unit = ||left_shoulder - right_shoulder||", style_body),
            Paragraph("Robustness across varying body statures and camera distances.", style_body)
        ],
        [
            Paragraph("Temporal Resampling", style_body),
            Paragraph("Linear interpolation to T = 64 frames", style_body),
            Paragraph("Standardizes variable trial lengths for batched GPU tensor computation.", style_body)
        ]
    ]
    t_prep = Table(prep_specs, colWidths=[120, 220, 196])
    t_prep.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_prep)
    story.append(Spacer(1, 2))

    # Embed Training Curves
    p_curves = Path("work/results/bigru/training_curves.png")
    if p_curves.exists():
        story.append(Image(str(p_curves), width=510, height=105))
        story.append(Paragraph("Figure 2.1: BiGRU baseline training loss and validation Macro-F1 progression across 35 epochs on full 60-word dataset.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>configs/config.yaml</code>, <code>signbangla/preprocess.py</code>, <code>scripts/06_preprocess.py</code>, <code>signbangla/dataset.py</code>, <code>scripts/08_test_dataloader.py</code>, <code>scripts/10_train_baseline.py</code>, <code>tests/test_pipeline.py</code>", style_card))

    # =========================================================================
    # PAGE 5: MEMBER 3 DETAILED ACCOMPLISHMENTS (SHESIR BAIDYA)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Group Member 3: Shesir Baidya (Real-Time System & Demo Lead)", style_page_heading))
    story.append(Paragraph("Role: Signer Splits, 30+ FPS Live Webcam HUD & Module 2 Speech/Text-to-Sign Engine", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=4))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 5)", style_h2))
    m3_bullets = (
        "<b>a) Signer-Independent Split Strategy (Zero Leakage):</b> Developed split generator (<code>scripts/07_make_splits.py</code>) "
        "enforcing strict zero-leakage subject isolation: 14 training signers (5,558 trials), 2 validation signers (U1, U11 -> 1,667 trials), "
        "and 2 held-out test signers (U4, U8 -> 1,060 trials). Exactly replicates the published benchmark evaluation protocol.<br/>"
        "<b>b) Real-Time Landmark Tracking & Latency Profiling:</b> Validated MediaPipe Holistic on live cameras (<code>scripts/02_test_mediapipe.py</code>). "
        "Discarding lower body and face mesh dropped inference latency by ~42%, maintaining 30+ FPS on standard Mac/PC hardware.<br/>"
        "<b>c) Interactive Live Webcam Demonstration (Module 1):</b> Engineered <code>scripts/11_live_webcam_demo.py</code> featuring "
        "rolling 64-frame sequence buffer, real-time shoulder normalization, model checkpoint autodetection (BiGRU or BiLSTM), "
        "and a bilingual HUD rendering predicted sign in English and native Bengali Unicode script (e.g., <i>Father / বাবা</i>), "
        "Top-3 probability bars, and sequence buffer gauge.<br/>"
        "<b>d) Module 2 Speech/Text-to-Sign Language Synthesis:</b> Built the reverse synthesis pipeline (<code>scripts/20_m2_demo.py</code>, "
        "<code>signbangla/speech2sign/</code>). Tokenizes input Bangla sentences, queries the 60-word clean sign dictionary, "
        "stitches consecutive gestures with smooth easing transitions (Cosine, Minimum-Jerk), and renders H.264 video with native Bangla typography.<br/>"
        "<b>e) Module 2 Test Suite Verification:</b> Formulated comprehensive unit tests (<code>tests/test_speech2sign.py</code>) "
        "validating dictionary lookup, trial ranking, easing interpolation, and font shaping &rarr; <b>15 passed in 2.2s</b>."
    )
    story.append(Paragraph(m3_bullets, style_body))
    story.append(Spacer(1, 2))

    story.append(Paragraph("B. Real-Time Demonstrator Previews: Module 1 & Module 2", style_h2))
    p_webcam = Path("work/mediapipe_test/webcam_demo_preview.png")
    p_m2 = Path("work/quality_plots/m2_demo_preview.png")

    demo_cells = []
    if p_webcam.exists() and p_m2.exists():
        demo_cells = [
            [
                Image(str(p_webcam), width=250, height=125),
                Image(str(p_m2), width=250, height=125)
            ],
            [
                Paragraph("<b>Figure 3.1 (Module 1):</b> Live Webcam Recognition HUD overlay displaying detected sign 'Father / বাবা' at 98.4% confidence with Top-3 probability meters at 31 FPS.", style_caption),
                Paragraph("<b>Figure 3.2 (Module 2):</b> Speech/Text-to-Sign animated synthesis (scripts/20_m2_demo.py) rendering stitched gesture 'বাবা আপেল' with smooth easing transitions and Bangla captions.", style_caption)
            ]
        ]
        t_demos = Table(demo_cells, colWidths=[265, 265])
        t_demos.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_demos)
    elif p_webcam.exists():
        story.append(Image(str(p_webcam), width=510, height=130))
        story.append(Paragraph("Figure 3.1: Live Webcam Recognition Demo HUD overlay displaying detected sign at 31 FPS.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>scripts/02_test_mediapipe.py</code>, <code>scripts/07_make_splits.py</code>, <code>signbangla/metrics.py</code>, <code>scripts/11_live_webcam_demo.py</code>, <code>scripts/20_m2_demo.py</code>, <code>scripts/20_build_sign_dictionary.py</code>, <code>tests/test_speech2sign.py</code>", style_card))

    # =========================================================================
    # PAGE 6: EXPERIMENTAL BENCHMARK RESULTS & IN-DEPTH DIAGNOSTIC ANALYSIS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Experimental Benchmark Results & Diagnostic Error Profiling", style_page_heading))
    story.append(Paragraph("Signer-Independent 60-Word Benchmark: BiGRU vs. BiLSTM Comparison, Confusion Matrix & Error Modes", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=0, spaceAfter=4))

    story.append(Paragraph("A. Quantitative Benchmark Performance Comparison", style_h2))
    eval_table_data = [
        [
            Paragraph("<b>Model Architecture & Split</b>", style_body_bold),
            Paragraph("<b>Top-1 Accuracy</b>", style_body_bold),
            Paragraph("<b>Macro F1</b>", style_body_bold),
            Paragraph("<b>Weighted F1</b>", style_body_bold),
            Paragraph("<b>Test Loss</b>", style_body_bold),
            Paragraph("<b>Parameters & Hardware</b>", style_body_bold)
        ],
        [
            Paragraph("<b>BiGRU + Attention (Test: U4, U8)</b>", style_body),
            Paragraph("<font color='#15803D'><b>86.79%</b> (920/1,060)</font>", style_body),
            Paragraph("<b>0.8353</b>", style_body),
            Paragraph("0.8590", style_body),
            Paragraph("<b>0.7138</b>", style_body),
            Paragraph("594,877 | Apple Silicon (MPS)", style_body)
        ],
        [
            Paragraph("<b>BiLSTM + Attention (Test: U4, U8)</b>", style_body),
            Paragraph("86.04% (912/1,060)", style_body),
            Paragraph("0.8293", style_body),
            Paragraph("0.8468", style_body),
            Paragraph("0.8120", style_body),
            Paragraph("759,741 | Apple Silicon (MPS)", style_body)
        ],
        [
            Paragraph("<b>Hasan et al. Benchmark (Test: U4, U8)</b>", style_body),
            Paragraph("75.10% (BiLSTM) / 67.6% (SVM)", style_body),
            Paragraph("—", style_body),
            Paragraph("—", style_body),
            Paragraph("—", style_body),
            Paragraph("Published Benchmark Baseline", style_body)
        ],
        [
            Paragraph("<b>BiGRU Validation Split (U1, U11)</b>", style_body),
            Paragraph("75.40% (1,257/1,667)", style_body),
            Paragraph("0.7369", style_body),
            Paragraph("0.7582", style_body),
            Paragraph("1.9587", style_body),
            Paragraph("Unseen Validation Signers", style_body)
        ]
    ]
    t_eval = Table(eval_table_data, colWidths=[150, 95, 60, 65, 55, 111])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 3))

    story.append(Paragraph("B. Test Confusion Matrix & Worst-Performing Classes", style_h2))
    p_cm = Path("work/results/bigru/test_confusion_matrix.png")

    # Worst Classes & Confused Pairs Table Data
    worst_classes_data = [
        [Paragraph("<b>Sign Class</b>", style_cell_bold), Paragraph("<b>Prec</b>", style_cell_bold), Paragraph("<b>Rec</b>", style_cell_bold), Paragraph("<b>F1</b>", style_cell_bold), Paragraph("<b>Primary Misclassification</b>", style_cell_bold)],
        [Paragraph("25: Chips", style_cell), Paragraph("0.0%", style_cell), Paragraph("0.0%", style_cell), Paragraph("<b>0.00</b>", style_cell), Paragraph("&rarr; 53: Capsule (83.3%)", style_cell)],
        [Paragraph("41: T-shirt", style_cell), Paragraph("11.1%", style_cell), Paragraph("8.3%", style_cell), Paragraph("<b>0.10</b>", style_cell), Paragraph("&rarr; 52: Bandage (91.7%)", style_cell)],
        [Paragraph("29: Cap", style_cell), Paragraph("28.6%", style_cell), Paragraph("33.3%", style_cell), Paragraph("<b>0.31</b>", style_cell), Paragraph("&rarr; 24: Sugar (50.0%)", style_cell)],
        [Paragraph("14: Mango", style_cell), Paragraph("100%", style_cell), Paragraph("21.4%", style_cell), Paragraph("<b>0.35</b>", style_cell), Paragraph("&rarr; 29: Cap (71.4%)", style_cell)],
        [Paragraph("37: Indebted", style_cell), Paragraph("60.0%", style_cell), Paragraph("27.3%", style_cell), Paragraph("<b>0.37</b>", style_cell), Paragraph("&rarr; 20: Jujube (72.7%)", style_cell)],
        [Paragraph("35: Cream", style_cell), Paragraph("83.3%", style_cell), Paragraph("33.3%", style_cell), Paragraph("<b>0.48</b>", style_cell), Paragraph("&rarr; 25: Chips (40.0%)", style_cell)],
        [Paragraph("0: Father", style_cell), Paragraph("100%", style_cell), Paragraph("47.4%", style_cell), Paragraph("<b>0.64</b>", style_cell), Paragraph("&rarr; 8: Grandmother (42.1%)", style_cell)],
    ]
    t_worst = Table(worst_classes_data, colWidths=[68, 38, 38, 38, 126])
    t_worst.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
    ]))

    diag_cells = [
        [
            Image(str(p_cm), width=210, height=210) if p_cm.exists() else Paragraph("Confusion matrix not found", style_body),
            [
                Paragraph("<b>Worst-Performing Classes & Primary Confusions:</b>", style_h2),
                t_worst,
                Spacer(1, 2),
                Paragraph("<b>Top Confused Pairs on Test Set:</b>", style_body_bold),
                Paragraph("• <b>T-shirt &harr; Bandage</b>: 11 misclassifications (91.7% of class)<br/>"
                          "• <b>Chips &harr; Capsule</b>: 10 misclassifications (83.3% of class)<br/>"
                          "• <b>Mango &harr; Cap</b>: 10 misclassifications (71.4% of class)<br/>"
                          "• <b>Indebted &harr; Jujube</b>: 8 misclassifications (72.7% of class)<br/>"
                          "• <b>Father &harr; Grandmother</b>: 8 misclassifications (42.1% of class)", style_card)
            ]
        ]
    ]
    t_diag = Table(diag_cells, colWidths=[220, 316])
    t_diag.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, 0), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_diag)
    story.append(Paragraph("Figure 4.1: Row-normalized 60x60 test confusion matrix on unseen benchmark signers U4 & U8 alongside worst-performing class breakdown.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<b>Kinematic & Visual Failure Mode Analysis:</b> With 60 vocabulary classes, error distribution is highly non-uniform: "
        "a small cluster of visually and kinematically similar signs accounts for the vast majority of errors, which provides significantly "
        "more actionable engineering insight than the 0.75-point gap between GRU and LSTM. Primary confusion patterns stem from: "
        "<b>(1) Identical Torso Arm-Wrapping Kinematics:</b> <i>T-shirt</i> and <i>Bandage</i> both involve rapid crossed forearms across the chest, "
        "where gross arm trajectories mask subtle finger flexion differences. "
        "<b>(2) Micro-Pinch Gestures Near the Mouth:</b> <i>Chips</i> and <i>Capsule</i> both rely on small index-thumb pinches held in front of the mouth, "
        "where monocular MediaPipe depth ambiguity (z-axis jitter) creates feature overlap. "
        "<b>(3) Near-Identical Stroke Trajectories:</b> <i>Father</i> and <i>Grandmother</i> both execute continuous chin-to-cheek strokes. "
        "Implementing Spatial-Temporal Graph Attention (ST-GCN) in Week 6 will directly resolve these localized finger joint ambiguities.",
        style_body
    ))

    # =========================================================================
    # PAGE 7: SYSTEM VERIFICATION, ROADMAP & PLAN FOR NEXT WEEK
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. System Verification, CLI Execution Guide & Plan for Next Week", style_page_heading))
    story.append(Paragraph("Integration Verification Checklist, Interactive Commands for Both Modules, and Week 6 Roadmap", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=0, spaceAfter=4))

    story.append(Paragraph("A. Full Pipeline Integration Verification Checklist", style_h2))
    test_data = [
        [Paragraph("<b>Verification Check</b>", style_body_bold), Paragraph("<b>Verification Method & Engineering Criteria</b>", style_body_bold), Paragraph("<b>Result</b>", style_body_bold)],
        [
            Paragraph("Module 1 Synthetic Pipeline Test", style_body),
            Paragraph("20 unit tests across synthetic data, annotations, resamplers, dataloaders & metrics.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED (20/20)</b></font>", style_body)
        ],
        [
            Paragraph("Module 2 Speech-to-Sign Test", style_body),
            Paragraph("15 unit tests across sign dictionary, easing transitions, motion stitching & rendering.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED (15/15)</b></font>", style_body)
        ],
        [
            Paragraph("Geometric Normalization", style_body),
            Paragraph("Preprocessed landmarks verify shoulder width = 1.0, shoulder midpoint = (0,0,0).", style_body),
            Paragraph("<font color='#15803D'><b>PASSED</b></font>", style_body)
        ],
        [
            Paragraph("BiGRU Baseline Training", style_body),
            Paragraph("35-epoch training on full 60-word BdSLW60 split &rarr; 86.79% Test Accuracy on U4 & U8.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED (86.79%)</b></font>", style_body)
        ],
        [
            Paragraph("BiLSTM Comparative Training", style_body),
            Paragraph("35-epoch training on exact same split &rarr; 86.04% Test Accuracy on U4 & U8.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED (86.04%)</b></font>", style_body)
        ],
        [
            Paragraph("Module 1 Live Webcam System", style_body),
            Paragraph("Live OpenCV feed with rolling 64-frame buffer, dual-model inference & bilingual HUD at 30+ FPS.", style_body),
            Paragraph("<font color='#15803D'><b>OPERATIONAL</b></font>", style_body)
        ],
        [
            Paragraph("Module 2 Motion Stitching & Video", style_body),
            Paragraph("Sentence tokenization, cosine easing transition stitching, and H.264 video rendering with Bangla text.", style_body),
            Paragraph("<font color='#15803D'><b>OPERATIONAL</b></font>", style_body)
        ]
    ]
    t_test = Table(test_data, colWidths=[140, 316, 80])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 2))

    # Prominent CLI Commands Box
    story.append(Paragraph("B. Interactive Execution Commands (How to Run & Test Both Modules)", style_h2))
    cmd_box = [
        [
            Paragraph("<b>1. Run Module 1 (Real-Time Live Webcam Recognition Demo):</b><br/>"
                      "<code>python scripts/11_live_webcam_demo.py --checkpoint work/results/bigru/best_model.pt</code><br/>"
                      "<i>(Auto-detects BiLSTM checkpoints via <code>--checkpoint work/results/bilstm/best_model.pt</code>, supports <code>--camera 0</code>)</i><br/><br/>"
                      "<b>2. Run Module 2 (Speech/Text-to-Sign Animation Video Synthesis):</b><br/>"
                      "<code>python scripts/20_m2_demo.py --text \"বাবা আপেল\" --n-transition 12 --easing cosine --out work/m2_demo.mp4</code><br/>"
                      "<i>(Stitches gestures with smooth easing curves and writes an H.264 video with native Bangla typography)</i><br/><br/>"
                      "<b>3. Run Module 2 Sign Dictionary Compilation & Unit Tests:</b><br/>"
                      "<code>python scripts/20_build_sign_dictionary.py && pytest tests/test_speech2sign.py</code>",
                      style_card)
        ]
    ]
    t_cmd = Table(cmd_box, colWidths=[536])
    t_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_cmd)
    story.append(Spacer(1, 2))

    story.append(Paragraph("C. Plan for Next Week (Week 6 Sprint Targets)", style_h2))
    next_week_plan = [
        [Paragraph("<b>Role / Member</b>", style_body_bold), Paragraph("<b>Target Task Description</b>", style_body_bold), Paragraph("<b>Deliverable Module</b>", style_body_bold)],
        [
            Paragraph("<b>Member 1: MD. Konok-Uz-Zaman</b><br/><font size=6 color='#64748B'>Data & Model Lead</font>", style_body),
            Paragraph("Implement <b>Spatial-Temporal Graph Convolution (ST-GCN)</b> and Transformer architectures to address localized finger pinch ambiguities.", style_body),
            Paragraph("<code>signbangla/models.py<br/>TransformerClassifier</code>", style_body)
        ],
        [
            Paragraph("<b>Member 2: Wahiduzzaman Mahin</b><br/><font size=6 color='#64748B'>Training & Evaluation Lead</font>", style_body),
            Paragraph("Execute systematic <b>ablation studies and hyperparameter tuning</b> (learning rate schedules, attention heads, focal loss for confused pairs).", style_body),
            Paragraph("<code>scripts/10_train_baseline.py<br/>work/results/tuning/</code>", style_body)
        ],
        [
            Paragraph("<b>Member 3: Shesir Baidya</b><br/><font size=6 color='#64748B'>Real-Time System & Demo Lead</font>", style_body),
            Paragraph("Implement <b>temporal confidence smoothing & sign boundary hysteresis</b> on the live webcam HUD; integrate microphone input for Module 2 speech.", style_body),
            Paragraph("<code>scripts/11_live_webcam_demo.py<br/>scripts/20_m2_demo.py</code>", style_body)
        ]
    ]
    t_next = Table(next_week_plan, colWidths=[140, 276, 120])
    t_next.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t_next)
    story.append(Spacer(1, 2))

    story.append(Paragraph("D. Master Development Roadmap (Weeks 6 through 10)", style_h2))
    story.append(Paragraph(
        "• <b>Week 6:</b> ST-GCN / Transformer sequence modeling, loss ablation for visually confused pairs, and temporal smoothing.<br/>"
        "• <b>Week 7:</b> Speech output integration (gTTS Bangla audio synthesis for Module 1) and microphone ASR integration for Module 2.<br/>"
        "• <b>Weeks 8–10:</b> Bidirectional system switch, usability testing with native signers, final benchmark report, and presentation.",
        style_body
    ))
    story.append(Spacer(1, 3))

    # Faculty Advisor Review Block
    advisor_box = [
        [
            Paragraph("<b>Faculty Advisor Review & Sign-Off:</b><br/>"
                      "Project: SignBangla (CSE299 Junior Design Project) &nbsp;|&nbsp; Faculty Advisor: Dr. Nabil Bin Hannan &nbsp;|&nbsp; Date: November 2026<br/>"
                      "Comments: __________________________________________________________________________________________<br/>"
                      "Signature: __________________________________________________ &nbsp;&nbsp;&nbsp;&nbsp; Status: [ &nbsp; ] Approved &nbsp; [ &nbsp; ] Revisions Requested",
                      style_card)
        ]
    ]
    t_adv = Table(advisor_box, colWidths=[536])
    t_adv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_adv)

    # Build the document
    doc.build(story, canvasmaker=NSUNumberedCanvas)
    print(f"Successfully generated {output_path}!")

if __name__ == "__main__":
    out_file = Path("SignBangla_Phase1_Progress_Presentation.pdf")
    create_presentation_pdf(out_file)
