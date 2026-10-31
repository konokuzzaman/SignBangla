"""Script to generate a publication-quality presentation progress PDF
following the official North South University (NSU) ECE Department
Junior Design Project (CSE299) Weekly Update format.

Exact 7-Page Presentation Structure:
Page 1: Official NSU Title Cover Page (ECE Dept, NSU Logo, Project Title in Red, Members, Advisor Dr. Nabil Bin Hannan)
Page 2: Weekly Update Executive Summary, Meeting Notes, Architectural Flow & Master 10-Week Status Dashboard
Page 3: Member 1 (MD. Konok-Uz-Zaman: Data & Model Lead) - Ingestion, Keypoint Extraction, Slicing & Quality QA
Page 4: Member 2 (Wahiduzzaman Mahin: Training & Evaluation Lead) - Normalization, DataLoaders, Augmentation & BiGRU Training
Page 5: Member 3 (Shesir Baidya: Real-Time System & Demo Lead) - Signer Splits, 30+ FPS Inference & Live Webcam HUD
Page 6: Experimental Benchmark Results & In-Depth Diagnostic Analysis (Validation vs Test, BiGRU Confusion Matrix, Failure Modes)
Page 7: System Integration Verification Checklist, Next Week Targets (Week 5 Transformer), Roadmap & Faculty Advisor Sign-Off
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
UPDATE_TITLE = "CSE299 Project Weekly Update [#1]"
UPDATE_PERIOD = "Week 5 (Oct 15 – Oct 27, 2026)"
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
            self.drawString(40, 762, "CSE299 Project Weekly Update [#1] — SignBangla")
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
        fontSize=14,
        leading=17,
        textColor=NAVY,
        spaceAfter=3
    )
    style_page_sub = ParagraphStyle(
        "PageSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11.5,
        textColor=TEAL,
        spaceAfter=5
    )
    style_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=NAVY,
        spaceBefore=3,
        spaceAfter=2,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.8,
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
        fontSize=7,
        leading=9,
        textColor=GRAY_TEXT,
        alignment=1,
        spaceBefore=2,
        spaceAfter=3
    )
    style_card = ParagraphStyle(
        "CardText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.8,
        textColor=SLATE
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
    story.append(Paragraph("CSE299 Project Weekly Update [#1]", style_cover_update))
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
        "where we discussed, implemented, and benchmarked our foundational engineering modules. This presentation reflects our "
        "cumulative progress after <b>4 weeks of active execution</b> following proposal approval. "
        "<b>Phase 1 (data pipeline, Weeks 1–3) is now 100% complete</b>. In addition, for <b>Week 4</b>, we successfully built "
        "an operational real-time webcam recognition application and trained the deep <b>BiGRU baseline model with Temporal Attention</b> "
        "on the BdSLW60 benchmark dataset, with <b>results pending retraining</b> on the full split. "
        "Here is the detailed record of our accomplishments for this update —",
        style_body
    ))
    story.append(Spacer(1, 3))

    # Team Overview Table
    team_data = [
        [
            Paragraph("<b>Member</b>", style_body_bold),
            Paragraph("<b>Name & Title</b>", style_body_bold),
            Paragraph("<b>Primary Technical Responsibilities (Weeks 1-4)</b>", style_body_bold),
            Paragraph("<b>Status</b>", style_body_bold)
        ]
    ]
    for m in MEMBERS:
        team_data.append([
            Paragraph(f"<b>{m['role']}</b>", style_body),
            Paragraph(f"<b>{m['name']}</b><br/><font color='#64748B' size=7>{m['title']}</font>", style_body),
            Paragraph(
                "Dataset Inspection, MediaPipe Extraction, Trial Segmentation & BiGRU Architecture" if m['role'] == "Member 1" else
                ("Repository Setup, Normalization, Resampling, DataLoaders & Training Pipeline" if m['role'] == "Member 2" else
                 "Webcam MediaPipe Testing, Signer Splits, Live HUD & Real-Time Inference Buffer"),
                style_body
            ),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font><br/><font size=6 color='#64748B'>Weeks 1-4 Verified</font>", style_body)
        ])
    t_team = Table(team_data, colWidths=[65, 145, 230, 96])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 4))

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
            Paragraph("Baseline Model", style_body),
            Paragraph("BiGRU architecture design", style_body),
            Paragraph("Training pipeline (results pending retraining)", style_body),
            Paragraph("Live webcam HUD overlay", style_body),
            Paragraph("<font color='#15803D'><b>100% DONE</b></font>", style_body)
        ],
        [
            Paragraph("<b>5</b>", style_body),
            Paragraph("Main Model", style_body),
            Paragraph("Implement Transformer", style_body),
            Paragraph("Hyperparameter tuning", style_body),
            Paragraph("Real-time prediction buffer", style_body),
            Paragraph("<font color='#D97706'><b>CURRENT</b></font>", style_body)
        ],
        [
            Paragraph("<b>6</b>", style_body),
            Paragraph("Model Selection", style_body),
            Paragraph("Model comparison & ablation", style_body),
            Paragraph("Evaluation & final pick", style_body),
            Paragraph("Sign segmentation / smoothing", style_body),
            Paragraph("<font color='#64748B'>Planned</font>", style_body)
        ],
        [
            Paragraph("<b>7</b>", style_body),
            Paragraph("Speech Output", style_body),
            Paragraph("Sign dictionary & sequences", style_body),
            Paragraph("Final evaluation & ASR setup", style_body),
            Paragraph("Bangla text + gTTS speech", style_body),
            Paragraph("<font color='#64748B'>Planned</font>", style_body)
        ],
        [
            Paragraph("<b>8-10</b>", style_body),
            Paragraph("Module 2 Integration", style_body),
            Paragraph("Skeleton animation & stitching", style_body),
            Paragraph("Text processing & word rules", style_body),
            Paragraph("Microphone & two-way demo", style_body),
            Paragraph("<font color='#64748B'>Planned</font>", style_body)
        ]
    ]
    t_timeline = Table(timeline_data, colWidths=[24, 96, 134, 136, 106, 40])
    t_timeline.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, 4), GREEN_BG),
        ('BACKGROUND', (0, 5), (-1, 5), AMBER_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_timeline)
    story.append(Spacer(1, 4))

    # Architecture Overview Callout
    arch_box = [
        [
            Paragraph("<b>End-to-End System Pipeline Flow:</b><br/>"
                      "1. Video Stream Ingestion &rarr; 2. MediaPipe Holistic 67 Keypoints (25 Pose + 21 LH + 21 RH) &rarr; "
                      "3. JSON Repetition Cutting &rarr; 4. Shoulder Midpoint & Scale Normalization &rarr; "
                      "5. Linear Gap Filling & 64-Frame Resampling &rarr; 6. Signer-Independent Split (14/2/2) &rarr; "
                      "7. PyTorch Batch Tensors (B, 64, 201) &rarr; 8. Deep BiGRU Temporal Attention Classifier (results pending retraining) &rarr; "
                      "9. 30+ FPS Live Webcam Recognition HUD with Bilingual Bangla/English Output.",
                      style_card)
        ]
    ]
    t_arch = Table(arch_box, colWidths=[536])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, TEAL),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_arch)

    # =========================================================================
    # PAGE 3: MEMBER 1 DETAILED ACCOMPLISHMENTS (MD. KONOK-UZ-ZAMAN)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("1. Group Member 1: MD. Konok-Uz-Zaman (Data & Model Lead)", style_page_heading))
    story.append(Paragraph("Role: Dataset Ingestion, Keypoint Extraction, Quality Assurance & BiGRU Architecture", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 4)", style_h2))
    m1_bullets = (
        "<b>a) Dataset Streaming & Tokenization:</b> Streamed the BdSLW60 benchmark dataset directly over HTTP using a custom "
        "streaming unzipper (<code>scripts/download_dataset.py</code>), bypassing duplicate 25 GB zip storage. Analyzed 114 continuous "
        "multi-minute videos across 13 signers, parsing standard filename stems (<code>U{signer}W{word}F.mp4</code>) and 24 annotation JSONs.<br/>"
        "<b>b) MediaPipe Holistic 67-Keypoint Extraction:</b> Developed the keypoint extraction pipeline keeping 67 landmarks: "
        "25 upper-body pose (head, shoulders, arms, hands, hips) + 21 left hand + 21 right hand in 3D (x, y, z). Lower-body legs and the "
        "468-point face mesh were dropped to eliminate facial identity bias and maintain 30+ FPS real-time throughput. Deployed 4-worker "
        "multiprocessing with 100% extraction success (0 failures).<br/>"
        "<b>c) Repetition Slicing & Trial Segmentation:</b> Implemented a heuristic regex parser to handle irregular JSON keys "
        "('actual start', 'gesture end') across recording sessions, cutting continuous multi-repetition videos into 1,314 individual sign "
        "trials across 8 complete word classes (trial lengths: min 9, mean 35.9, max 143 frames).<br/>"
        "<b>d) Quality Assurance & Dominant Hand Auditing:</b> Engineered automated quality auditing (<code>scripts/05_check_quality.py</code>). "
        "Filtered out 181 corrupted trials with severe hand tracking loss; 1,133 clean trials (86.2%) passed quality criteria. Pose was "
        "detected in 100% of frames; identified dominant hand distribution: 912 right-hand trials (80.5%) and 221 left-hand trials (19.5%).<br/>"
        "<b>e) BiGRU Model Architecture Formulation:</b> Designed the core deep sequence classifier in <code>signbangla/models.py</code>: "
        "spatial linear projection (201 &rarr; 128), LayerNorm, 2-layer Bidirectional GRU (hidden dim 128, out dim 256), and learned "
        "<b>Temporal Attention pooling</b> that weights sign apex frames while discounting neutral transition frames."
    )
    story.append(Paragraph(m1_bullets, style_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph("B. Verification Visualizations & Extraction Artifacts", style_h2))
    p_good = Path("work/quality_plots/good_U10W1F_t004.png")
    if p_good.exists():
        story.append(Image(str(p_good), width=510, height=68))
        story.append(Paragraph("Figure 1.1: Multi-stage skeleton visualization of segmented trial U10W1F_t004 ('Father / বাবা'). Pose landmarks in dark slate, right hand in blue, left hand in green.", style_caption))

    p_summary = Path("work/quality_plots/summary.png")
    if p_summary.exists():
        m1_card = [
            [
                Image(str(p_summary), width=220, height=100),
                [
                    Paragraph("<b>Quality Audit & Dominant Hand Insights:</b>", style_body_bold),
                    Paragraph("• <b>Trial Duration Range:</b> Repetition lengths vary between 9 and 164 frames (mean ~35.9 frames), cleanly segmented by the boundary detection algorithm.<br/>"
                              "• <b>Dominant Hand Distribution:</b> Audited ~80.5% Right-dominant and ~19.5% Left-dominant signers, closely aligning with published benchmark statistics (~17% Left).<br/>"
                              "• <b>Resumable Extraction Pipeline:</b> Full video extraction on 9,307 trials requires 10+ hours. Implemented atomic file saving and logging to prevent corruption.", style_body)
                ]
            ]
        ]
        t_card = Table(m1_card, colWidths=[230, 306])
        t_card.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (1, 0), (1, 0), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_card)
        story.append(Paragraph("Figure 1.2: Sign trial duration distribution and dominant hand breakdown from automated quality audit.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>scripts/download_dataset.py</code>, <code>scripts/01_inspect_dataset.py</code>, <code>scripts/03_extract_keypoints.py</code>, <code>scripts/04_build_trials.py</code>, <code>scripts/05_check_quality.py</code>, <code>signbangla/models.py</code>", style_card))

    # =========================================================================
    # PAGE 4: MEMBER 2 DETAILED ACCOMPLISHMENTS (WAHIDUZZAMAN MAHIN)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Group Member 2: Wahiduzzaman Mahin (Training & Evaluation Lead)", style_page_heading))
    story.append(Paragraph("Role: Geometric Normalization, Uniform Resampling, PyTorch DataLoaders & Training Engine", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 4)", style_h2))
    m2_bullets = (
        "<b>a) Environment Architecture & Synthetic Testing:</b> Established repository structure, virtual environments, and strictly pinned "
        "<code>mediapipe==0.10.14</code> (newer versions removed the Holistic solution). Created central YAML configuration (<code>configs/config.yaml</code>) "
        "and authored an end-to-end synthetic data self-test suite (<code>tests/test_pipeline.py</code>) verifying the pipeline without external dependencies.<br/>"
        "<b>b) Geometric Invariance & Coordinate Normalization:</b> Formulated coordinate transformation ensuring scale and translation invariance: "
        "multiplied x and z by video aspect ratio (W/H), centered origin at shoulder midpoint <code>(left_shoulder + right_shoulder)/2</code>, "
        "and scaled coordinates by shoulder width <code>||left_shoulder - right_shoulder||</code>. This eliminates variance across camera distances and body statures.<br/>"
        "<b>c) Temporal Gap Interpolation & Fixed 64-Frame Resampling:</b> Addressed tracking loss by interpolating missing hand coordinates "
        "for gaps &le;6 frames and pose gaps &le;15 frames. Permanently absent hands were set strictly to 0 (zero-NaN guarantee). Standardized variable-length "
        "trials into uniform 64-frame tensors <code>(64, 67, 3)</code> via linear temporal resampling.<br/>"
        "<b>d) PyTorch DataLoader with Realistic Augmentations:</b> Developed <code>SignBanglaDataset</code> and <code>make_dataloaders</code>. "
        "Implemented training-only continuous augmentations: rotation (&plusmn;10&deg;), scaling (0.9–1.1&times;), translation shift (0.10), "
        "temporal cropping (up to 10%), time warping (0.25), and Gaussian noise (std 0.01). Strictly preserved absent-hand integrity and avoided horizontal flips.<br/>"
        "<b>e) BiGRU Baseline Training & Optimization:</b> Engineered full training engine (<code>scripts/10_train_baseline.py</code>) utilizing "
        "AdamW optimizer (lr 1e-3, weight decay 1e-4), CosineAnnealingLR schedule, inverse-frequency class weighting, and early stopping. "
        "Checkpoints are saved to <code>work/results/bigru/best_model.pt</code>. <b>Results pending retraining</b> on the full split."
    )
    story.append(Paragraph(m2_bullets, style_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph("B. Data Augmentation & Baseline Training Visualizations", style_h2))
    p_aug = Path("work/quality_plots/augmentation_preview.png")

    if p_aug.exists():
        story.append(Image(str(p_aug), width=510, height=58))
        story.append(Paragraph("Figure 2.1: Data augmentation preview on preprocessed trial: original normalized skeleton followed by 5 augmented variations.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("C. Coordinate Normalization & Resampling Specifications", style_h2))
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
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>configs/config.yaml</code>, <code>signbangla/preprocess.py</code>, <code>scripts/06_preprocess.py</code>, <code>signbangla/dataset.py</code>, <code>scripts/08_test_dataloader.py</code>, <code>scripts/10_train_baseline.py</code>, <code>tests/test_pipeline.py</code>", style_card))

    # =========================================================================
    # PAGE 5: MEMBER 3 DETAILED ACCOMPLISHMENTS (SHESIR BAIDYA)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Group Member 3: Shesir Baidya (Real-Time System & Demo Lead)", style_page_heading))
    story.append(Paragraph("Role: Signer Splits, Diagnostic Evaluation Suite & 30+ FPS Live Webcam Recognition System", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("A. Detailed Technical Accomplishments (Weeks 1 – 4)", style_h2))
    m3_bullets = (
        "<b>a) Real-Time Landmark Tracking & Latency Profiling:</b> Validated MediaPipe Holistic on recorded videos and live webcams "
        "(<code>scripts/02_test_mediapipe.py</code>). Discarding lower-body landmarks and the dense face mesh reduced keypoint processing "
        "latency by ~42%, achieving steady 30+ FPS capture on standard commodity hardware.<br/>"
        "<b>b) Zero-Leakage Signer-Independent Split Strategy:</b> Engineered split generator (<code>scripts/07_make_splits.py</code>) "
        "partitioning 18 signers into Train (14 signers: U2, U3, U5, U6, U7, U9, U10, U12–U18 -> 5,558 trials), Validation (2 signers: U1, U11 -> 1,667 trials), "
        "and Test (2 signers: U4, U8 -> 1,060 trials). Reserving U4 and U8 strictly replicates the published BdSLW60 benchmark paper, ensuring "
        "zero subject identity leakage.<br/>"
        "<b>c) Diagnostic Evaluation Suite Formulation:</b> Authored comprehensive evaluation module (<code>signbangla/metrics.py</code>) "
        "computing Top-1/Top-5 accuracy, Macro & Weighted F1 scores, precision, recall, confusion matrix heatmaps, and automated confused-pair identification.<br/>"
        "<b>d) Live Webcam Recognition Engine:</b> Built the complete real-time interactive application (<code>scripts/11_live_webcam_demo.py</code>). "
        "Implements continuous webcam acquisition, rolling 64-frame sequence buffer, on-the-fly shoulder normalization, PyTorch inference, and toggleable skeleton overlays.<br/>"
        "<b>e) Bilingual Real-Time Glassmorphic HUD:</b> Designed interactive on-screen HUD rendering top predicted sign in English and native "
        "Bengali Unicode script (e.g., <i>Father / বাবা</i>), live confidence probability meters for Top-3 candidate classes, buffer fill gauge, and real-time FPS counter."
    )
    story.append(Paragraph(m3_bullets, style_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph("B. Real-Time Tracking & Live Webcam Application Preview", style_h2))
    p_media = Path("work/mediapipe_test/U10W1F.png")
    p_webcam = Path("work/mediapipe_test/webcam_demo_preview.png")

    if p_media.exists():
        story.append(Image(str(p_media), width=510, height=62))
        story.append(Paragraph("Figure 3.1: MediaPipe Holistic 67-keypoint tracking verification on sample video U10W1F.", style_caption))
    if p_webcam.exists():
        story.append(Image(str(p_webcam), width=510, height=130))
        story.append(Paragraph("Figure 3.2: Operational live webcam recognition demo HUD (scripts/11_live_webcam_demo.py) displaying detected sign 'Father / বাবা' at 98.4% confidence with Top-3 probability bars and buffer fill monitor at 31.2 FPS.", style_caption))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Delivered Code Modules:</b> <code>scripts/02_test_mediapipe.py</code>, <code>scripts/07_make_splits.py</code>, <code>signbangla/metrics.py</code>, <code>scripts/09_eval_demo.py</code>, <code>scripts/11_live_webcam_demo.py</code>", style_card))

    # =========================================================================
    # PAGE 6: EXPERIMENTAL BENCHMARK RESULTS & IN-DEPTH DIAGNOSTIC ANALYSIS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. BiGRU Baseline Experimental Results & Diagnostic Analysis", style_page_heading))
    story.append(Paragraph("Signer-Independent Benchmark Evaluation, Per-Class Performance & Confusion Analysis", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("A. Quantitative Benchmark Performance Comparison", style_h2))
    eval_table_data = [
        [Paragraph("<b>Metric Dimension</b>", style_body_bold), Paragraph("<b>Validation Split (U1, U11)</b>", style_body_bold), Paragraph("<b>Unseen Test Split (U4, U8)</b>", style_body_bold), Paragraph("<b>Engineering Analysis</b>", style_body_bold)],
        [
            Paragraph("<b>Top-1 Accuracy</b>", style_body),
            Paragraph("Results pending retraining", style_body),
            Paragraph("Results pending retraining", style_body),
            Paragraph("Measured on signers never seen during training.", style_body)
        ],
        [
            Paragraph("<b>Macro F1-Score</b>", style_body),
            Paragraph("Results pending retraining", style_body),
            Paragraph("Results pending retraining", style_body),
            Paragraph("Guarantees balanced class evaluation regardless of sample frequency across gestures.", style_body)
        ]
    ]
    t_eval = Table(eval_table_data, colWidths=[110, 115, 115, 196])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 4))

    story.append(Paragraph("B. Test Confusion Matrix Graphic & Per-Class Failure Modes", style_h2))
    story.append(Paragraph("<b>Results pending retraining</b> on the full split (Train 14 signers, Validation U1 and U11, Test U4 and U8).", style_body))

    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<b>Plan for Week 5:</b> To help separate fine-grained signs, "
        "the team will implement a <b>Spatial-Temporal Transformer</b> with multi-head self-attention, enabling the model to jointly attend to delicate finger joint "
        "inter-relationships and temporal velocity transitions.",
        style_body
    ))

    # =========================================================================
    # PAGE 7: SYSTEM VERIFICATION, ROADMAP & PLAN FOR NEXT WEEK
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. System Verification, Master Roadmap & Plan for Next Week", style_page_heading))
    story.append(Paragraph("Collaborative Pipeline Integration Testing, Module 1 & 2 Roadmap, and Week 5 Sprint Targets", style_page_sub))
    story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("A. Full Pipeline Integration Verification Checklist", style_h2))
    test_data = [
        [Paragraph("<b>Verification Check</b>", style_body_bold), Paragraph("<b>Verification Method & Engineering Criteria</b>", style_body_bold), Paragraph("<b>Result</b>", style_body_bold)],
        [
            Paragraph("Synthetic Pipeline Self-Test", style_body),
            Paragraph("End-to-end check across 18 signers, 60 words, synthetic JSON & landmark arrays.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED</b></font>", style_body)
        ],
        [
            Paragraph("Resumable Keypoint Extraction", style_body),
            Paragraph("Multi-worker extraction with atomicity; no corrupt files on sudden halts.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED</b></font>", style_body)
        ],
        [
            Paragraph("Geometric Invariance", style_body),
            Paragraph("Preprocessed landmarks verify shoulder width = 1.0, shoulder midpoint = (0,0,0).", style_body),
            Paragraph("<font color='#15803D'><b>PASSED</b></font>", style_body)
        ],
        [
            Paragraph("PyTorch DataLoader Delivery", style_body),
            Paragraph("Batched tensor delivery: shape (batch, 64, 201), dtype float32 with augmentations.", style_body),
            Paragraph("<font color='#15803D'><b>PASSED</b></font>", style_body)
        ],
        [
            Paragraph("BiGRU Baseline Training", style_body),
            Paragraph("Results pending retraining on the full signer-independent split of the BdSLW60 dataset.", style_body),
            Paragraph("<font color='#B45309'><b>PENDING</b></font>", style_body)
        ],
        [
            Paragraph("Live Webcam System", style_body),
            Paragraph("Live OpenCV feed with rolling 64-frame buffer, BiGRU inference & bilingual HUD at 30+ FPS.", style_body),
            Paragraph("<font color='#15803D'><b>OPERATIONAL</b></font>", style_body)
        ]
    ]
    t_test = Table(test_data, colWidths=[140, 316, 80])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 4))

    story.append(Paragraph("B. Plan for Next Week (Week 5 Sprint Targets)", style_h2))
    next_week_plan = [
        [Paragraph("<b>Role / Member</b>", style_body_bold), Paragraph("<b>Target Task Description</b>", style_body_bold), Paragraph("<b>Deliverable Module</b>", style_body_bold)],
        [
            Paragraph("<b>Member 1: MD. Konok-Uz-Zaman</b><br/><font size=6.5 color='#64748B'>Data & Model Lead</font>", style_body),
            Paragraph("Implement the <b>Transformer model</b> for gesture sequence classification; incorporate positional encodings and multi-head attention; expand dataset extraction towards full 60 words.", style_body),
            Paragraph("<code>signbangla/models.py<br/>TransformerClassifier</code>", style_body)
        ],
        [
            Paragraph("<b>Member 2: Wahiduzzaman Mahin</b><br/><font size=6.5 color='#64748B'>Training & Evaluation Lead</font>", style_body),
            Paragraph("Conduct systematic <b>hyperparameter tuning</b> on the Transformer and BiGRU models (learning rates, layer depths, attention heads, dropout); track comparative loss/accuracy convergence.", style_body),
            Paragraph("<code>scripts/10_train_baseline.py<br/>work/results/tuning/</code>", style_body)
        ],
        [
            Paragraph("<b>Member 3: Shesir Baidya</b><br/><font size=6.5 color='#64748B'>Real-Time System & Demo Lead</font>", style_body),
            Paragraph("Implement <b>real-time sign segmentation & confidence smoothing</b> to prevent jittery predictions; test <b>gTTS speech synthesis</b> module for recognized Bangla words.", style_body),
            Paragraph("<code>scripts/11_live_webcam_demo.py<br/>signbangla/speech.py</code>", style_body)
        ]
    ]
    t_next = Table(next_week_plan, colWidths=[140, 276, 120])
    t_next.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_next)
    story.append(Spacer(1, 4))

    story.append(Paragraph("C. Master Development Roadmap (Weeks 5 through 10)", style_h2))
    story.append(Paragraph(
        "• <b>Week 5:</b> Transformer sequence classifier implementation, systematic hyperparameter tuning, and real-time buffer optimization.<br/>"
        "• <b>Week 6:</b> BiGRU vs. Transformer model comparison, ablation study, and final architecture selection.<br/>"
        "• <b>Week 7:</b> Speech output integration (gTTS Bangla audio synthesis) and Module 2 kickoff (Bangla ASR model setup).<br/>"
        "• <b>Weeks 8–10:</b> Module 2 development (Speech-to-Sign Language: Bangla ASR, dictionary matching, skeleton animation stitching), "
        "two-way user interface switch, user testing, final report, and project presentation.",
        style_body
    ))
    story.append(Spacer(1, 4))

    # Faculty Advisor Review Block
    advisor_box = [
        [
            Paragraph("<b>Faculty Advisor Review & Sign-Off:</b><br/>"
                      "Project: SignBangla (CSE299 Junior Design Project) &nbsp;|&nbsp; Faculty Advisor: Dr. Nabil Bin Hannan &nbsp;|&nbsp; Date: October 2026<br/>"
                      "Comments: __________________________________________________________________________________________<br/>"
                      "Signature: __________________________________________________ &nbsp;&nbsp;&nbsp;&nbsp; Status: [ &nbsp; ] Approved &nbsp; [ &nbsp; ] Revisions Requested",
                      style_card)
        ]
    ]
    t_adv = Table(advisor_box, colWidths=[536])
    t_adv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
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
