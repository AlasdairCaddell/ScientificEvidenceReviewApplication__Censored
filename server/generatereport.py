import os
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from pageobj import Document
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from pageobj import Document

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

import scraper

def checklisttype1(item):
    if type(item) == list:
        return item[0]
    else:
        return item


def create_pdf_report(reportdict):
   # try:
        # Create a canvas object
        filename = os.path.join(reportdict["storepage"], reportdict["name"] + "report.pdf")
        pdf_canvas = canvas.Canvas(filename, pagesize=letter)

        # Set the font and font size
        pdf_canvas.setFont("Helvetica", 11)

        # Add the document name as the title
        pdf_canvas.drawCentredString(4.25 * inch, 10.5 * inch, reportdict["name"])

        # Add the subject of the paper (most common entities and variables)
        entities_text = ", ".join(reportdict["entities"])
        variables_text = ", ".join(reportdict["variables"])
        
        
        pdf_canvas.drawString(1 * inch, 10 * inch, f"Common Entities: {entities_text}")
        pdf_canvas.drawString(1 * inch, 9.75 * inch, f"Common Subjects: {variables_text}")

        # Add a line separator
        pdf_canvas.line(1 * inch, 9.5 * inch, 7.5 * inch, 9.5 * inch)
        pdf_canvas.drawString(1 * inch, 9.25 * inch, "Rules:")
        pdf_canvas.drawString(3 * inch, 9.25 * inch, "Similarity and Occurrence")

        y = 9 * inch
        line_height = 0.25 * inch
        
        max_width = 5.5 * inch
        for arr in reportdict["similarities"]:
            ruletext=arr[0]
            textlocation=arr[1]
            score=arr[2]
                    # Write information into columns
            
            if y - line_height < 4:
                # If not, start a new page
                pdf_canvas.showPage()
                # Reset the y-coordinate for the new page
                y = 10 * inch
                
                
            pdf_canvas.drawString(1 * inch, y, f"Most Likely Location: Page{textlocation}")
            pdf_canvas.drawString(5 * inch, y, f"Similarity: {score}")

            rule_lines = textwrap.wrap(ruletext, width=60)  # Adjust the width based on your layout
            for line in rule_lines:
                if y - line_height < 4:
                # If not, start a new page
                    pdf_canvas.showPage()
                    # Reset the y-coordinate for the new page
                    y = 10 * inch
                pdf_canvas.drawString(1 * inch, y - line_height, line)
                
                y -= line_height
            
            # Calculate the number of lines needed for the rule text
            #lines_needed = pdf_canvas.drawText(ruletext, 1.25 * inch, y)

            # Move to the next line(s)
            #y -= lines_needed * line_height     


            y -= line_height

        # Add a line separator
            pdf_canvas.line(1 * inch, y, 7.5 * inch, y)
            y -= 0.25 * inch

        # Add the reference section header
        pdf_canvas.drawString(1 * inch, y, "References:")
        pdf_canvas.drawString(6 * inch, y, "Citation Count")

        y -= 0.25 * inch
        for reference in reportdict["references"]:
            
            if y - line_height < 4:
                # If not, start a new page
                pdf_canvas.showPage()
                # Reset the y-coordinate for the new page
                y = 10 * inch
            
            
            if reference['title']:
                #print(reference['title'])
                reference_line = textwrap.wrap(checklisttype1(reference['title']), width=60)  # Adjust the width based on your layout
                for refline in reference_line:
                    if y - line_height < 4:
                    # If not, start a new page
                        pdf_canvas.showPage()
                        # Reset the y-coordinate for the new page
                        y = 10 * inch
                    pdf_canvas.drawString(1 * inch, y - line_height, refline)
                    
                    y -= line_height
                #pdf_canvas.drawString(1 * inch, y, reference['title'])
            else:
                reference_line = textwrap.wrap(checklisttype1(reference['raw_ref']), width=60)  # Adjust the width based on your layout
                for refline in reference_line:
                    if y - line_height < 4:
                    # If not, start a new page
                        pdf_canvas.showPage()
                        # Reset the y-coordinate for the new page
                        y = 10 * inch
                    pdf_canvas.drawString(1 * inch, y - line_height, refline)
                    
                    y -= line_height
                
                #pdf_canvas.drawString(1 * inch, y, reference['raw_ref'])

            # Get citation count
            citation = scraper.scrape_one_google_scholar_page(checklisttype1(reference['raw_ref']))
            if (citation == -2):
                citation="Too many requests"
            elif citation == -1:
                citation = "Not found"
            
            time.sleep(5)
            citation_text = f"{citation}" 
            pdf_canvas.drawString(6 * inch, y, citation_text)

            y -= 0.25 * inch
           # pdf_canvas.drawString(1 * inch, y, f"link: {reference['url']}")
            y -= 0.25 * inch
            
        pdf_canvas.save()
        return filename
    #except Exception as e:
    #    print(e)
  #      return None
#
#data=


if __name__=="__main__":   
    data = {
    'name': 'Matchingguidelinequaltest',
    'storepage': 'e:\\Lvl4Project\\ScientificEvidenceReviewApplication\\server\\storePDF\\Matchingguidelinequal',
    'entities': ['quantum', '#', '1'],
    'variables': ['quantum', 'particle', 'research'],
    'similarities': [
        ['identify key issues/topic consideration .', 6, 0.20511396931923237],
        ['acknowledge funding source contributor .', 2, 0.3113321006420896],
        ['acknowledge conflict interest~ .', 2, 0.4783958082375246],
        ['state problem/question/objectives investigation .', 4, 0.19960427795143879],
        ['indicate study design , including type participant data source , analytic strategy , main results/findings , main implications/significance .', 3, 0.398328404572064],
        ['keywords~ .', 4, 0.10780974180000809],
        ['frame problem question context .', 3, 0.3012523613927557],
        ['review , critique , synthesize applicable literature identify key issues/debates/theoretical framework relevant literature clarify barrier , knowledge gap , practical need .', 3, 0.5541669148758906],
        ['state purpose , goal , and/or aim study .', 3, 0.3569928278049862],
        ['state target audience~ .', 6, 0.19601383591278412],
        ['provide rationale fit design used investigate purpose/goal .', 3, 0.37554375394488065],
        ['describe approach inquiry , illuminates objective research rationale .', 3, 0.35337068617424494],
        ['summarize research design , including data-collection strategy , data-analytic strategy , , illuminating , approach inquiry .', 3, 0.40978047195066286],
        ['provide rationale design selected .', 3, 0.38793315992752336],
        ['describe researcher ` background approaching study , emphasizing prior understanding phenomenon study .', 3, 0.4829882942298519],
        ['describe prior understanding phenomenon study managed and/or influenced research .', 6, 0.37484922544219446],
        ['provide number participants/documents/events analyzed .', 5, 0.18620181607698605],
        ['describe demographics/cultural information , perspective participant , characteristic data source might influence data collected .', 3, 0.25496367449760393],
        ['describe existing data source .', None, 0.28024043693638556],
        ['provide data repository information openly shared data~ .', 5, 0.21966710150330476],
        ['describe archival search process locating data analyses~ .', 4, 0.33835560743812293],
        ['describe relationship interaction researcher participant relevant research process impact research process .', 3, 0.47744112662375676],
        ['describe recruitment process recruitment protocol .', 4, 0.5960577690057136],
        ['describe incentive compensation , provide assurance relevant ethical process data collection consent process relevant .', 4, 0.3659049517998602],
        ['describe process number participant determined relation study design .', 3, 0.3461114419433189],
        ['provide change number attrition final number participants/sources .', 4, 0.17656816626855446],
        ['describe rationale decision halt data collection .', 3, 0.3283164679333078],
        ['convey study purpose portrayed participant , different purpose stated .', 4, 0.38207696658050705],
        ['describe participants/data source selection process inclusion/exclusion criterion .', 4, 0.28810084781917333],
        ['provide general context study .', 6, 0.2947324007826354],
        ['participant selection archived data set , describe recruitment selection process data set well decision selecting set participant data set .', 4, 0.44222399307949745],
        ['state form data collected .', 4, 0.2925953326916809],
        ['describe origin evolution data-collection protocol .', 4, 0.27992067427831363],
        ['describe alteration data-collection strategy response evolving finding study rationale .', 5, 0.4422831773064556],
        ['describe data-selection data-collection process .', 3, 0.15851166205565995],
        ['convey extensiveness engagement .', 2, 0.2098140812080576],
        ['interview written study , indicate mean range time duration data-collection process .', 4, 0.33569504814177364],
        ['describe management use reflexivity data-collection process , illuminates study .', 3, 0.31903027199706396],
        ['describe question asked data collection : content central question , form question .', 4, 0.3621250944954306],
        ['identify data audio/visual recording method , field note , transcription process used .', None, 0.37071815848134193],
        ['describe method procedure used purpose goal .', 3, 0.3145065961107538],
        ['explicate detail process analysis , including discussion procedure following principle transparency .', None, 0.45524420077212713],
        ['describe coder analyst training~ ( already described researcher description section ) .', 3, 0.4262328856812681],
        ['identify whether coding category emerged analysis developed priori .', 5, 0.4704188117933154],
        ['identify unit analysis unit formed~ .', 5, 0.418367365483657],
        ['describe process arriving analytic scheme~ .', None, 0.32903597674189133],
        ['provide illustration description analytic scheme development~ .', 5, 0.2972408464475813],
        ['indicate software~ .', 6, 0.0834433686010248],
        ['demonstrate claim made analysis warranted produced finding methodological integrity . procedure support methodological integrity typically described across relevant section paper , could addressed separate section elaboration emphasis would helpful . issue methodological integrity include following : ass adequacy data term ability capture form diversity relevant question , research goal , inquiry approach .', 3, 0.5294373748418142],
        ['demonstrate claim made analysis warranted produced finding methodological integrity . procedure support methodological integrity typically described across relevant section paper , could addressed separate section elaboration emphasis would helpful . issue methodological integrity include following : describe researcher ` perspective managed data collection analysis .', 3, 0.504747763035293],
        ['demonstrate claim made analysis warranted produced finding methodological integrity . procedure support methodological integrity typically described across relevant section paper , could addressed separate section elaboration emphasis would helpful . issue methodological integrity include following : demonstrate finding grounded evidence .', 3, 0.46808763900166767],
        ['demonstrate claim made analysis warranted produced finding methodological integrity . procedure support methodological integrity typically described across relevant section paper , could addressed separate section elaboration emphasis would helpful . issue methodological integrity include following : demonstrate contribution insightful meaningful .', 3, 0.44463143668785654],
        ['demonstrate claim made analysis warranted produced finding methodological integrity . procedure support methodological integrity typically described across relevant section pa finding methodological integrity . procedure support methodological integrity typically described across relevant section paper , could addressed separate section elaboration emphasis would helpful . issue methodological integrity include following : present finding coherent manner make sense contradiction disconfirming evidence data', 3, 0.499213270566657],
        ['demonstrate consistency regard analytic process describe response inconsistency , relevant .', 5, 0.5797145498447005],
        ['describe support claim supplemented check added qualitative analysis . example supplemental check strengthen research may include transcripts/data collected returned participant feedback', 6, 0.3471620477314367],
        ['describe support claim supplemented check added qualitative analyntal check strengthen research may include check interview thoroughness interviewer demand', 5, 0.39842243047833187],
        ['describe support claim supplemented check added qualitative analysis . example supplemental check strengthen research may include consensus auditing process', 5, 0.4575124023237288],
        ['describe support claim supplemented check added qualitative analysis . example supplemental check strengthen research may include member check participant feedback finding', 5, 0.35839249991049477],
        ['desc supplemental check strengthen research may include in-depth thick description , case example , illustration', 3, 0.3666989533930158],
        ['describe support claim supplemented check added qualitative analysis . example supplemental check strengthen research may include structured method researcher reflexivity', 3, 0.42558032306498916],
        ['describe support claim supplemented check added qualitative analysis . example supplemental check strengthen research may include check utility finding responding study probible study design .', 6, 0.44297675958591537],
        ['present synthesizing illustration , useful organizing conveying finding .', None, 0.3656474770248778],
        ['describe central contribution significance advancing disciplinary understanding .', 5, 0.349522601244365],
        ['describe type contribution made finding finding best utilized .', 6, 0.323437316427806],
        ['identify similarity difference prior theory research finding .', 6, 0.7479172194302798],
        ['reflect alternative explanation finding .', 6, 0.6953830613822574],
        ['identify study ` strength limitation', 6, 0.5539882065605867],
        ['describe limit scope transferability', 6, 0.31991674069238624],
        ['revisit ethical dilemma challenge encountered , provide related suggestion future researcher .', 6, 0.5917034702727093],
        ['consider implication future research , policy , practice .', 6, 0.4954158141418612]],
    'references' : [
        {
        "raw_ref": "Einstein, A., Podolsky, B., & Rosen, N. (1935). \"Can Quantum-Mechanical Description of Physical Reality be Considered Complete?\" Physical Review, 47(10), 777-780.",
        "authors": ["Einstein, A.", "Podolsky, B.", "Rosen, N."],
        "year": 1935,
        "title": "Can Quantum-Mechanical Description of Physical Reality be Considered Complete?",
        "journal": "Physical Review",
        "volume": 47,
        "issue": 10,
        "pages": "777-780"
    },
    {
        "raw_ref": "Feynman, R. P. (1982). \"Simulating Physics with Computers.\" International Journal of Theoretical Physics, 21(6/7), 467-488.",
        "authors": ["Feynman, R. P."],
        "year": 1982,
        "title": "Simulating Physics with Computers",
        "journal": "International Journal of Theoretical Physics",
        "volume": 21,
        "issue": "6/7",
        "pages": "467-488"
    },
    {
        "raw_ref": "Hawking, S. (1988). \"A Brief History of Time.\" Bantam Books.",
        "authors": ["Hawking, S."],
        "year": 1988,
        "title": "A Brief History of Time",
        "publisher": "Bantam Books"
    },
    {
        "raw_ref": "Schrödinger, E. (1935). \"Discussion of Probability Relations between Separated Systems.\" Proceedings of the Cambridge Philosophical Society, 31(4), 555-563.",
        "authors": ["Schrödinger, E."],
        "year": 1935,
        "title": "Discussion of Probability Relations between Separated Systems",
        "journal": "Proceedings of the Cambridge Philosophical Society",
        "volume": 31,
        "issue": 4,
        "pages": "555-563"
    },
    {
        "raw_ref": "Nielsen, M. A., & Chuang, I. L. (2010). \"Quantum Computation and Quantum Information.\" Cambridge University Press.",
        "authors": ["Nielsen, M. A.", "Chuang, I. L."],
        "year": 2010,
        "title": "Quantum Computation and Quantum Information",
        "publisher": "Cambridge University Press"
    },
    {
        "raw_ref": "Bell, J. S. (1964). \"On the Einstein-Podolsky-Rosen Paradox.\" Physics, 1(3), 195-200.",
        "authors": ["Bell, J. S."],
        "year": 1964,
        "title": "On the Einstein-Podolsky-Rosen Paradox",
        "journal": "Physics",
        "volume": 1,
        "issue": 3,
        "pages": "195-200"
    },
    {
        "raw_ref": "IBM Quantum. (2023). \"Advancements in Quantum Computing.\" IBM Research Blog. [Online]. Available: https://www.ibm.com/blogs/research/2023/04/advancements-in-quantum-computing/",
        "authors": ["IBM Quantum"],
        "year": 2023,
        "title": "Advancements in Quantum Computing",
        "source": "IBM Research Blog",
        "url": "https://www.ibm.com/blogs/research/2023/04/advancements-in-quantum-computing/"
    },
    {
        "raw_ref": "Nature Physics. (2022). \"Special Issue: Quantum Information and Quantum Computing.\" Nature Physics, 18(2), 111-224.",
        "authors": ["Nature Physics"],
        "year": 2022,
        "title": "Special Issue: Quantum Information and Quantum Computing",
        "journal": "Nature Physics",
        "volume": 18,
        "issue": 2,
        "pages": "111-224"
    }]}
    create_pdf_report(data)