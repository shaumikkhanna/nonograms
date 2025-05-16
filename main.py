import pickle
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import cidfonts
from pathlib import Path
import random


font_path = Path("./DejaVuSansMono.ttf")  # Place it in a 'fonts' folder
pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(font_path)))



def make_nonogram_pdfs(nonogram_list, cols=5, rows=8, title="Nonogram Sheet", description=""):
    PAGE_WIDTH, PAGE_HEIGHT = A4
    MARGIN = 1.5 * cm
    TITLE_SPACE = 3.5 * cm  # increased to allow description
    GRID_WIDTH = PAGE_WIDTH - 2 * MARGIN
    GRID_HEIGHT = PAGE_HEIGHT - MARGIN - TITLE_SPACE

    cell_width = GRID_WIDTH / cols
    cell_height = GRID_HEIGHT / rows

    per_page = cols * rows
    total_pages = (len(nonogram_list) + per_page - 1) // per_page

    for page_num in range(total_pages):
        if total_pages > 1:
            c = canvas.Canvas(f"{title}_{page_num + 1}.pdf", pagesize=A4)
        else:
            c = canvas.Canvas(f"{title}.pdf", pagesize=A4)

        # Title
        c.setFont("Helvetica-Bold", 16)
        if total_pages > 1:
            c.drawString(MARGIN, PAGE_HEIGHT - 1.5 * cm, f"{title} - Page {page_num + 1}")
        else:
            c.drawString(MARGIN, PAGE_HEIGHT - 1.5 * cm, title)

        # Description
        c.setFont("Helvetica", 12)
        c.drawString(MARGIN, PAGE_HEIGHT - 2.2 * cm, description)

        # Grid content
        c.setFont("DejaVuSansMono", 10)

        for i in range(per_page):
            index = page_num * per_page + i
            if index >= len(nonogram_list):
                break

            row = i // cols
            col = i % cols

            x = MARGIN + col * cell_width
            y = PAGE_HEIGHT - TITLE_SPACE - row * cell_height

            # Draw cell border
            c.rect(x, y - cell_height, cell_width, cell_height)

            nonogram_str = nonogram_list[index]
            lines = nonogram_str.split('\n')
            line_height = 10
            total_text_height = len(lines) * line_height
            start_y = y - (cell_height / 2) + (total_text_height / 2)
            padding_x = cell_width * 0.3

            for line_idx, line in enumerate(lines):
                c.drawString(x + padding_x, start_y - line_idx * line_height, line)

        c.save()


with open("pickles/output_4x4d.pkl", "rb") as file:
    data = pickle.load(file)
    

# nonogram_list = [d['final_state'] for d in data[0]]
# make_nonogram_pdfs(nonogram_list, title="4x4 Undetermined", description=f"These {len(nonogram_list)} nonograms were such that no contradiction or unique solution was found.")
# nonogram_list = [d['final_state'] for d in data[-1]]
# make_nonogram_pdfs(nonogram_list, title="4x4 Uniquely Solved", description=f"These {len(nonogram_list)} nonograms were such that a unique solution was found.")
# nonogram_list = [d['final_state'] for d in data[1]]
# make_nonogram_pdfs(nonogram_list, title="4x4 Unsolvable", description=f"These {len(nonogram_list)} nonograms were such that a contradiction was found making them unsolvable.")

# nonogram_list = [d['final_state'] for d in data[0]]
# make_nonogram_pdfs(nonogram_list, title="4x4 Undetermined", description=f"These {len(nonogram_list)} nonograms were such that no contradiction or unique solution was found.")

# nonogram_list = []
# for d in random.sample(data[0], 100):
#     nonogram_list.append(d['final_state'])
# make_nonogram_pdfs(nonogram_list, cols=4, rows=7, title="5x5 Undetermined", description=f"These are some nonograms out of 3128 were such that no contradiction or unique solution was found.")

# nonogram_list = [d['final_state'] for d in data]
# make_nonogram_pdfs(nonogram_list, cols=4, rows=6, title="6x6 Undetermined_100", description=f"These are some 6x6 nonograms were such that no contradiction or unique solution was found.")

# nonogram_list = [d['final_state'] for d in data]
# make_nonogram_pdfs(nonogram_list, cols=3, rows=6, title="7x7 Undetermined_100", description=f"These are some 6x6 nonograms were such that no contradiction or unique solution was found.")


