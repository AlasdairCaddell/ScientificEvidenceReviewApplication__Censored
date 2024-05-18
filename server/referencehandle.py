import re
import scraper
import cleantext

#from refextract import extract_journal_reference

def cleant(text):
    return cleantext.clean(text)


def emptydef(text):
    cleant(text)
#name,name,
#[3] Shi Y, Huang Z, Chen J, Deng M, Su M, Qi J, et al. Fine-grained ZnO ceramic fabricated by high-pressure cold 
#    sintering. Ceramics International. 2022.
    
    
    refnumberpattern = r'\[(\d+)\]'
    refnumberpattern2 = r'\d+'
    #\[\d+\]\s[A-Z][a-z]+\s[A-Z][a-z]+(?:,\s[A-Z]\.)?(?:,\s?[A-Z]\.?[A-Z]?\.)?(?:,\s?[A-Z]\.?[A-Z]?\.)?\s(et al\.)?\s[A-Z][a-z]+\s[A-Z]\.\s(?:[A-Z][a-z]+,?\s?)*\d{4};\d{2}:\d{4}-\d{2}

    nameregex = r'^(?:\s*{?([A-Za-z]+(?:\s[A-Za-z]+)?(?:,\s[A-Za-z])*)}?\s*)?(?:\"(.*)\")?\s*\n?\s*{([A-Za-z0-9\s]+)}?\s*\n?\s*{?((?:[0-9]{1,3},?)?[0-9]{1,3})}?\s*(?:\((?:19|20)[0-9]{2}\))?\s*\n?\s*{?([0-9]{1,4}-[0-9]{1,4})}?\s*$'
    websiteregex=r'^\s*www\.\w+\.com/([\w-]+)/?'
    titleregex=r'([A-Z][\w\s]+[.?!]*)'
    dateregex= r'[\d]{4,}[\d:;-]*\.' 
    
    
    
    
    #edgecase issues with titles that end in year, do last
    #\b\d{4}([.;:]\d+)?([:;-]\d+)?([:;-]\d+)?\d+\b
    # 
    # re_pattern = re.compile('.+?(\d+)\.([a-zA-Z0-9+])')
    # 2020;580:730-9.
    
    regex=r"\[([0-9]+)\]\s([A-Z][a-z.]+\s?[A-Z]?\.?,\s?.*?)\.\s([A-Z][a-z]+\s?[A-Z]?\.?,?\s?.*?)\.\s([A-Za-z]+\s?[A-Za-z]*:?[\d-]*);?([\d-]*)?:?([\d-]*)?\.?"
    
    apajournal=r"(?:^|\s)([A-Z][a-z]+, [A-Z]?[a-z]?,?\s?\(?[A-Z][a-z]+\)?\.?(?:\s|\.)*)\(([12]\d{3}[a-z]?)\)\. ([A-Za-z\s]+)\. (.*)(?:\.\s|\.$|$)"
    
    
    
    
    
    
    tempregex=r"^(?P<author>[A-Z](?:(?!$)[A-Za-z\s&.,'`])+)\((?P<year>\d{4})\)\.?\s*(?P<title>[^()]+?[?.!])\s*(?:(?:(?P<journal>(?:(?!^[A-Z])[^.]+?)),\s*(?P<issue>\d+)[^,.]*(?=,\s*\d+|.\s*Ret))|(?:In\s*(?P<editors>[^()]+))\(Eds?\.\),\s*(?P<book>[^().]+)|(?:[^():]+:[^().]+\.)|(?:Retrieved|Paper presented))"

    #textarr=ref.split(delimiter)
    
    
    #for text in textarr:
    
    numbers_in_brackets = re.search(tempregex, text.replace("\n", ""))
    re.compile("^(?P<author>[A-Z](?:(?!$)[A-Za-z\s&.,'`])+)\((?P<year>\d{4})\)\.?\s*(?P<title>[^()]+?[?.!])\s*(?:(?:(?P<journal>(?:(?!^[A-Z])[^.]+?)),\s*(?P<issue>\d+)[^,.]*(?=,\s*\d+|.\s*Ret))|(?:In\s*(?P<editors>[^()]+))\(Eds?\.\),\s*(?P<book>[^().]+)|(?:[^():]+:[^().]+\.)|(?:Retrieved|Paper presented))")
    
    #ref = re.findall(dateregex, text)   
    #reference_match = re.search(reference_pattern, text)

    #if reference_match:
    #reference = reference_match.group()  # Get the matched reference
    #leftover_text = text[reference_match.end():]
    
    return numbers_in_brackets



if __name__ == '__main__':

    t1="""[1] Ndayishimiye A, Sengul MY, Bang SH, Tsuji K, Takashima K, de Beauvoir TH, et al. Comparing hydrothermal sintering and cold sintering process: Mechanisms, microstructure, kinetics and chemistry. Journal of the European Ceramic Society. 2020;40:1312-24. [2] Guo J, Si M, Zhao X, Wang L, Wang K, Hao J, et al. Altering interfacial properties through the integration of C60 into ZnO ceramic via cold sintering process. Carbon. 2022.
    [3] Shi Y, Huang Z, Chen J, Deng M, Su M, Qi J, et al. Fine-grained ZnO ceramic fabricated by high-pressure cold 
    sintering. Ceramics International. 2022.
    [4] Guo J, Floyd R, Lowum S, Maria J-P, Herisson de Beauvoir T, Seo J-H, et al. Cold sintering: progress, challenges, 
    and future opportunities. Annual Review of Materials Research. 2019;49.
    [5] Galotta A, Sglavo VM. The cold sintering process: A review on processing features, densification mechanisms 
    and perspectives. Journal of the European Ceramic Society. 2021;41:1-17.
    [6] Nur K, Mishra TP, da Silva JGP, Gonzalez-Julian J, Bram M, Guillon O. Influence of powder characteristics on 
    cold sintering of nano-sized ZnO with density above 99%. Journal of the European Ceramic Society. 2021;41:2648-
    62.
    [7] Jing Y, Luo N, Wu S, Han K, Wang X, Miao L, et al. Remarkably improved electrical conductivity of ZnO 
    ceramics by cold sintering and post-heat-treatment. Ceramics International. 2018;44:20570-4.
    [8] Gao J, Ding Q, Yan P, Liu Y, Huang J, Mustafa T, et al. Highly Improved Microwave Absorbing and Mechanical 
    Properties in Cold Sintered ZnO by Incorporating Graphene Oxide. Journal of the European Ceramic Society. 
    2022;42:993-1000.
    [9] Bang SH, Tsuji K, Ndayishimiye A, Dursun S, Seo JH, Otieno S, et al. Toward a size scale‐up cold sintering 
    process at reduced uniaxial pressure. Journal of the American Ceramic Society. 2020;103:2322-7.
    [10] Liang J, Zhao X, Kang S, Guo J, Chen Z, Long Y, et al. Microstructural evolution of ZnO via hybrid cold 
    sintering/spark plasma sintering. Journal of the European Ceramic Society. 2022;42:5738-46.
    [11] Suleiman B, Zhang H, Ding Y, Li Y. Microstructure and mechanical properties of cold sintered porous alumina 
    ceramics. Ceramics International. 2022;48:13531-40.
    [12] Lowum S, Floyd R, Bermejo R, Maria J-P. Mechanical strength of cold-sintered zinc oxide under biaxial bending. 
    Journal of materials science. 2019;54:4518-22.
    [13] Nur K, Zubair M, Gibson JS-L, Sandlöbes-Haut S, Mayer J, Bram M, et al. Mechanical properties of cold sintered 
    ZnO investigated by nanoindentation and micro-pillar testing. Journal of the European Ceramic Society. 2022;42:512-
    24.
    [14] Li L, Yan H, Hong WB, Wu SY, Chen XM. Dense gypsum ceramics prepared by room-temperature cold sintering 
    with greatly improved mechanical properties. Journal of the European Ceramic Society. 2020;40:4689-93.
    [15] Essa F, Zhang Q, Huang X, Ibrahim AMM, Ali MKA, Abdelkareem MA, et al. Improved friction and wear of M50 steel composites incorporated with ZnO as a solid lubricant with different concentrations under different loads. Journal of Materials Engineering and Performance. 2017;26:4855-66.
    [16] Essa F, Zhang Q, Huang X, Ali MKA, Elagouz A, Abdelkareem MA. Effects of ZnO and MoS2 solid lubricants 
    on mechanical and tribological properties of M50-steel-based composites at high temperatures: experimental and 
    simulation study. Tribology Letters. 2017;65:1-29.
    [17] Mobarhan G, Zolriasatein A, Ghahari M, Jalili M, Rostami M. The enhancement of wear properties of compressor 
    oil using MoS2 nano-additives. Advanced Powder Technology. 2022;33:103648.
    [18] Mousavi SB, Heris SZ, Estellé P. Viscosity, tribological and physicochemical features of ZnO and MoS2 diesel 
    oil-based nanofluids: An experimental study. Fuel. 2021;293:120481.
    [19] Mousavi SB, Heris SZ, Estellé P. Experimental comparison between ZnO and MoS2 nanoparticles as additives 
    on performance of diesel oil-based nano lubricant. Scientific reports. 2020;10:1-17.
    [20] Chouhan A, Sarkar TK, Kumari S, Vemuluri S, Khatri OP. Synergistic lubrication performance by 
    incommensurately stacked ZnO-decorated reduced graphene oxide/MoS2 heterostructure. Journal of colloid and 
    interface science. 2020;580:730-9.
    [21] Ren B, Gao L, Li M, Zhang S, Ran X. Tribological properties and anti-wear mechanism of ZnO@ graphene core-shell nanoparticles as lubricant additives. Tribology International. 2020;144:106114.
    [22] Zhou X, Wang K, Wu Y, Wang X, Zhang X. Mussel-Inspired Interfacial Modification for Ultra-Stable MoS2 
    Lubricating Films with Improved Tribological Behavior on Nano-Textured ZnO Surfaces Using the AACVD Method. 
    ACS Applied Materials & Interfaces. 2022.
    [23] Elsheikh AH, Yu J, Sathyamurthy R, Tawfik M, Shanmugan S, Essa F. Improving the tribological properties of 
    AISI M50 steel using Sns/Zno solid lubricants. Journal of Alloys and Compounds. 2020;821:153494.
    [24] Yang Y, Zhao Y, Mei H, Cheng L, Zhang L. 3DN C/SiC-MoS2 self-lubricating composites with high friction 
    stability and excellent elevated-temperature lubrication. Journal of the European Ceramic Society. 2021;41:6815-23.
    [25] Su Y, Zhang Y, Song J, Hu L. Novel approach to the fabrication of an alumina-MoS2 self-lubricating composite 
    via the in situ synthesis of nanosized MoS2. ACS applied materials & interfaces. 2017;9:30263-6.
    [26] Staudacher M, Lube T, Schlacher J, Supancic P. Comparison of biaxial strength measured with the Ball-on-Three-Balls-and the Ring-on-Ring-test. Open Ceramics. 2021;6:100101.
    [27] Gruber M, Kraleva I, Supancic P, Danzer R, Bermejo R. A novel approach to assess the mechanical reliability of 
    thin, ceramic-based multilayer architectures. Journal of the European Ceramic Society. 2020;40:4727-36.
    [28] Shaly AA, Priya GH, Mahendiran M, Linet JM, Mani JAM. An intrinsic analysis on the nature of alumina 
    (Al2O3) reinforced hydroxyapatite nanocomposite. Physica B: Condensed Matter. 2022;642:414100.
    [29] Yoshimura H, Molisani AL, Narita N, Manholetti J, Cavenaghi J. Mechanical properties and microstructure of 
    zinc oxide varistor ceramics. Materials science forum: Trans Tech Publ; 2006. p. 408-13.   edgecase---------------------------------------------------------------------------------------------------------------------
    [30] Colas G, Serles P, Saulot A, Filleter T. Strength measurement and rupture mechanisms of a micron thick 
    nanocrystalline MoS2 coating using AFM based micro-bending tests. Journal of the Mechanics and Physics of Solids. 
    2019;128:151-61.
    [31] Alves MFRP, Dos Santos C, Elias CN, Amarante JEV, Ribeiro S. Comparison between different fracture 
    toughness techniques in zirconia dental ceramics. Journal of Biomedical Materials Research Part B: Applied 
    Biomaterials. 2022.
    [32] Moradkhani A, Baharvandi H, Tajdari M, Latifi H, Martikainen J. Determination of fracture toughness using the 
    area of micro-crack tracks left in brittle materials by Vickers indentation test. Journal of Advanced Ceramics. 
    2013;2:87-102.
    [33] Roy TK. Estimation of fracture toughness in ZnO ceramics from indentation crack opening displacement 
    measurements. Measurement. 2019;137:588-94.
    [34] Roy TK. Assessing hardness and fracture toughness in sintered zinc oxide ceramics through indentation 
    technique. Materials Science and Engineering: A. 2015;640:267-74.
    [35] Wang X, Tabarraei A, Spearot DE. Fracture mechanics of monolayer molybdenum disulfide. Nanotechnology. 
    2015;26:175703.
    [36] Kanthavel K, Sumesh K, Saravanakumar P. Study of tribological properties on Al/Al2O3/MoS2 hybrid composite 
    processed by powder metallurgy. Alexandria Engineering Journal. 2016;55:13-7.
    [37] Quan X, Zhang S, Hu M, Gao X, Jiang D, Sun J. Tribological properties of WS2/MoS2-Ag composite films 
    lubricated with ionic liquids under vacuum conditions. Tribology International. 2017;115:389-96.
    [38] Rouhi M, Moazami-Goudarzi M, Ardestani M. Comparison of effect of SiC and MoS2 on wear behavior of Al 
    matrix composites. Transactions of Nonferrous Metals Society of China. 2019;29:1169-83.
    [39] Wasekar NP, Bathini L, Ramakrishna L, Rao DS, Padmanabham G. Pulsed electrodeposition, mechanical 
    properties and wear mechanism in Ni-W/SiC nanocomposite coatings used for automotive applications. Applied 
    Surface Science. 2020;527:146896.
    [40] Gou J, Sun M, Yao J, Lin J, Liu J, Wang Y, et al. A Comparison Study of the Friction and Wear Behavior of 
    Nanostructured Al2O3-YSZ Composite Coatings With and Without Nano-MoS2. Journal of Thermal Spray 
    Technology. 2022:1-14. 
    [41] Staszuk M, Pakuła D, Reimann Ł, Król M, Basiaga M, Mysłek D, et al. Structure and properties of ZnO coatings 
    obtained by atomic layer deposition (ALD) method on a Cr-Ni-Mo steel substrate type. Materials. 2020;13:4223.
    [42] Hu KH, Hu XG, Wang J, Xu YF, Han CL. Tribological properties of MoS2 with different morphologies in high-density polyethylene. Tribology Letters. 2012;47:79-90."""  
    
    t2="""
    22. van der Horst K, Mathias KC, Patron AP, Allirot X. Art on a plate: a pilot evaluation of an international initiative designed to promote consumption of fruits and vegetables by children. J Nutr Educ Behav. 2019;51(8):919-925.e1. https://doi.org/10.1016/j.jneb.2019.03.009.
23. McMorrow L, Ludbrook A, Macdiarmid JI, Olajide D. Perceived barriers towards healthy eating and their association with fruit and vegetable consumption. J Public Health. 2016. https://doi.org/10.1093/pubmed/fdw038.
24. Olson CM. Behavioral nutrition interventions using e- and m-health communication technologies: a narrative review. Annu Rev Nutr. 
2016;36(1):647–64. https://doi.org/10.1146/annurev-nutr-071715-050815.
25. Vasiloglou MF, Marcano I, Lizama S, Papathanail I, Spanakis EK, Mougiakakou S. Multimedia data-based mobile applications for dietary 
assessment. J Diabetes Sci Technol. 2022. https://doi.org/10.1177/19322968221085026.
26. Azevedo LB, Stephenson J, Ells L, et al. The efectiveness of e-health interventions for the treatment of overweight or obesity in children 
and adolescents: a systematic review and meta-analysis. Obes Rev. 2022. https://doi.org/10.1111/obr.13373.
27. Burrows T, Hutchesson M, Chai LK, Rollo M, Skinner G, Collins C. Nutrition interventions for prevention and management of childhood 
obesity: what do parents want from an eHealth program? Nutrients. 2015;7(12):10469–79. https://doi.org/10.3390/nu7125546.
28. Chen Y, Perez-Cueto F, Giboreau A, Mavridis I, Hartwell H. The promotion of eating behaviour change through digital interventions. Int J 
Environ Res Public Health. 2020;17(20):7488. https://doi.org/10.3390/ijerph17207488.
29. Vasiloglou MF, Christodoulidis S, Reber E, et al. Perspectives and preferences of adult smartphone users regarding nutrition and diet apps: 
web-based survey study. JMIR mHealth uHealth. 2021;9(7):e27885. https://doi.org/10.2196/27885.
30. Hutchesson M, Gough C, Müller AM, et al. eHealth interventions targeting nutrition, physical activity, sedentary behavior, or obesity in 
adults: a scoping review of systematic reviews. Obes Rev. 2021. https://doi.org/10.1111/obr.13295.
31. Kankanhalli A, Shin J, Oh H. Mobile-based interventions for dietary behavior change and health outcomes: scoping review. JMIR mHealth 
uHealth. 2019;7(1):e11312. https://doi.org/10.2196/11312.
32. Burgess-Champoux T, Marquart L, Vickers Z, Reicks M. Perceptions of children, parents, and teachers regarding whole-grain foods, and 
implications for a school-based intervention. J Nutr Educ Behav. 2006;38(4):230–7. https://doi.org/10.1016/j.jneb.2006.04.147.
33. Garaulet M, Pérez-Llamas F, Rueda CM, Zamora S. Trends in the mediterranean diet in children from south-east Spain. Nutr Res. 
1998;18(6):979–88. https://doi.org/10.1016/S0271-5317(98)00081-5.
34. Kantor LS, Variyam JN, Allshouse JE, Putnam JJ, Lin BH. Choose a variety of grains daily, especially whole grains: a challenge for consumers. 
J Nutr. 2001;131(2):473S-486S. https://doi.org/10.1093/jn/131.2.473S.
35. Meynier A, Chanson-Rollé A, Riou E. Main factors infuencing whole grain consumption in children and adults—a narrative review. Nutrients. 2020;12(8):2217. https://doi.org/10.3390/nu12082217.
36. Tharner A, Jansen PW, Kiefte-de Jong JC, et al. Toward an operative diagnosis of fussy/picky eating: a latent profle approach in a population-based cohort. Int J Behav Nutr Phys Act. 2014;11(1):14. https://doi.org/10.1186/1479-5868-11-14.
37. Cano SC, Tiemeier H, van Hoeken D, et al. Trajectories of picky eating during childhood: a general population study. Int J Eat Disord. 
2015;48(6):570–9. https://doi.org/10.1002/eat.22384.
38. Dean M, O’Kane C, Issartel J, et al. Guidelines for designing age-appropriate cooking interventions for children: the development of 
evidence-based cooking skill recommendations for children, using a multidisciplinary approach. Appetite. 2021;161:105125. https://
doi.org/10.1016/j.appet.2021.105125.
39. Berner Fachhochschule. Kids Cooking@Home; 2022. https://www.bfh.ch/de/forschung/referenzprojekte/kids-cooking-home/. Accessed 
20 Oct 2022.
40. Infomaniak. Infomaniak - The Ethical Cloud. https://www.infomaniak.com/en; 2022. Accessed 20 Oct 2022.
41. Flutter. Flutter - Build apps for any screen. https://futter.dev; 2022. Accessed 20 Oct2022.
42. Spring.io, 2022. https://spring.io. Accessed 20 Oct 2022.
43. Spring boot, 2022. https://spring.io/projects/spring-boot. Accessed 20 Oct 2022.
44. My SQL, 2022. https://www.mysql.com/. Accessed 20 Oct 2022.
45. Appleton KA, Hemingway A, Rajska J, Hartwell H. Repeated exposure and conditioning strategies for increasing vegetable liking and 
intake: systematic review and meta-analyses of the published literature. Am J Clin Nutr. 2018;108(4):842–56. https://doi.org/10.1093/
ajcn/nqy143.
46. van der Horst K, Ferrage A, Rytz A. Involving children in meal preparation. Efects on food intake. Appetite. 2014;79:18–24. https://doi.
org/10.1016/j.appet.2014.03.030.
47. Maiz E, Urkia-Susin I, Urdaneta E, Allirot X. Child involvement in choosing a recipe, purchasing ingredients, and cooking at school increases 
willingness to try new foods and reduces food neophobia. J Nutr Educ Behav. 2021;53(4):279–89. https://doi.org/10.1016/j.jneb.2020.12.
015.
48. Allirot X, Da Quinta N, Chokupermal K, Urdaneta E. Involving children in cooking activities: a potential strategy for directing food choices 
toward novel foods containing vegetables. Appetite. 2016;103:275–85. https://doi.org/10.1016/j.appet.2016.04.031.
49. Saxe-Custack A, Egan S. Flint families cook: a virtual cooking and nutrition program for families. J Nutr Educ Behav. 2022;54(4):359–63. 
https://doi.org/10.1016/j.jneb.2022.01.002.
50. Khazen W, Jeanne JF, Demaretz L, Schäfer F, Fagherazzi G. Rethinking the use of mobile apps for dietary assessment in medical research. 
J Med Internet Res. 2020;22(6):e15619. https://doi.org/10.2196/15619.
51. Lu Y, Stathopoulou T, Vasiloglou MF, et al. goFOODTM: an artifcial intelligence system for dietary assessment. Sensors. 2020;20(15):4283. 
https://doi.org/10.3390/s20154283.
52. Vasiloglou MF, Christodoulidis S, Reber E, et al. What healthcare professionals think of “nutrition & diet” apps: an international survey. 
Nutrients. 2020;12(8):2214. https://doi.org/10.3390/nu12082214]."""
    #print(t2)
    #print(extract_journal_reference(t2))
    numref=emptydef(t1)
    print(numref)
   # print(len(ref))
#import re

# Sample text
#text = "[15] Essa F, Zhang Q, Huang X, Ibrahim AMM, Ali MKA, Abdelkareem MA, et al. Improved friction and wear..."

# First regex expression to match the reference number
#reference_pattern = r'\[\d+\]'
#reference_match = re.search(reference_pattern, text)

#if reference_match:
   # reference = reference_match.group()  # Get the matched reference
   # leftover_text = text[reference_match.end():]  # Get the leftover text after the reference

    # Second regex expression to match author names
 #   author_pattern = r"[A-Z][a-z]+\s[a-z]*[A-Z-]+[a-z]*\,"
 #   author_match = re.match(author_pattern, leftover_text)

 #   if author_match:
  #      authors = author_match.group()  # Get the matched authors
  #      remaining_text = leftover_text[author_match.end():]  # Get the remaining text after the authors

   #     print("Reference:", reference)
   #     print("Authors:", authors)
  #      print("Remaining Text:", remaining_text)
 #   else:
#        print("No matching authors found.")
#else:
 #   print("No matching reference found.")
