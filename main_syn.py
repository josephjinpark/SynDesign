import os, sys, re, regex, torch, pickle, gzip
from itertools import cycle

import multiprocessing as mp
from collections import defaultdict

from scipy import stats
from sklearn.preprocessing import minmax_scale



import numpy as np
import pandas as pd

from genet import database as db
import genet.utils
from genet.utils import reverse_complement as revcom
from genet.utils import cVCFData

from genet import predict as prd
from genet import design
from genet import models

## system config ##
sBASE_DIR = os.getcwd()

os.environ['MPLCONFIGDIR'] = '%s/images/matplt_tmp' % sBASE_DIR
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


NUM_CPUs  = os.cpu_count()
NUM_GPUs  = torch.cuda.device_count()

CORES = 1
ALT_INDEX = 60  # 60nts --- Alt --- 60nts *0-based
bTEST = 0  # 0 or 1



class cCOSMIC:
    def __init__(self):
        self.sGeneName = ''
        self.sAccID = ''
        self.nCDSLen = 0
        self.sHGCNID = ''  # SKIP for now
        self.sSample = ''  # SKIP for now
        self.sSampleID = ''  # SKIP for now
        self.sTumorID = ''  # SKIP for now
        self.sPriSite = ''  # primary site  ex) pancreas
        self.sSiteSub1 = ''  # SKIP for now
        self.sSiteSub2 = ''  # SKIP for now
        self.sSiteSub3 = ''  # SKIP for now
        self.sPriHist = ''  # primary histology
        self.sHistSub1 = ''  # SKIP for now
        self.sHistSub2 = ''  # SKIP for now
        self.sHistSub3 = ''  # SKIP for now
        self.bGenomeWide = ''  # ex) y or n
        self.sMutaID = ''  # SKIP for now
        self.sAltType = ''  # ex) c.35G>T
        self.sRef = ''
        self.sAlt = ''
        self.sAAType = ''  # ex) p.G12V
        self.sMutaDescri = ''  # ex) Substitution - Missense
        self.sMutaZygo = ''  # SKIP for now
        self.bLOH = ''  # loss of heterzygosity ex) y or n
        self.sGRCh = ''  # Genome Version
        self.sGenicPos = ''  # 17:7673781-7673781
        self.nChrID = ''  # 17  X = 24 Y = 25
        self.sChrID = ''  # chr17 or chrX and chrY
        self.sPos = ''  # 7673781   1-based
        self.sStrand = ''
        self.bSNP = ''  # ex) y and n
        self.sDelete = ''  # ex) PATHOGENIC
    # def END : __init__
def parse_cosmic(cosmic_file):

    dict_out = {}
    if cosmic_file.endswith('gz'):
        inf = gzip.open(cosmic_file, 'rt', encoding='utf-8')

    else:
        inf = open(cosmic_file, 'r')

    for i, line in enumerate(inf):

        if line.startswith('Gene'): continue  # SKIP HEADER

        list_sColumn = line.strip('\n').split('\t')
        '''
        if i == 0:
            list_sHeader = list_sColumn
        elif i == 1:
            for i,(a,b) in enumerate(zip(list_sHeader, list_sColumn)):
                print('%s\t%s\t%s' % (i,a,b))
        else: break

        '''
        cos             = cCOSMIC()
        cos.sGeneName   = list_sColumn[0].upper()
        cos.sAccID      = list_sColumn[1]
        cos.nCDSLen     = int(list_sColumn[2])
        cos.sHGCNID     = list_sColumn[3]
        cos.sSample     = list_sColumn[4]
        cos.sSampleID   = list_sColumn[5]
        cos.sTumorID    = list_sColumn[6]
        cos.sPriSite    = list_sColumn[7]
        cos.sSiteSub1   = list_sColumn[8]
        cos.sSiteSub2   = list_sColumn[9]
        cos.sSiteSub3   = list_sColumn[10]
        cos.sPriHist    = list_sColumn[11]
        cos.sHistSub1   = list_sColumn[12]
        cos.sHistSub2   = list_sColumn[13]
        cos.sHistSub3   = list_sColumn[14]
        cos.bGenomeWide = True if list_sColumn[15] == 'y' else False
        cos.sMutaID     = list_sColumn[16]
        cos.sAltType    = list_sColumn[17]
        cos.sAAType     = list_sColumn[18]
        cos.sMutaDescri = list_sColumn[19]
        cos.sMutaZygo   = list_sColumn[20]
        cos.bLOH        = True if list_sColumn[21] == 'y' else False
        cos.sGRCh       = list_sColumn[22]
        cos.sGenicPos   = list_sColumn[23]
        if not list_sColumn[23]: continue  # Skip those w/o position information

        cos.nChrID = list_sColumn[23].split(':')[0]

        if cos.nChrID not in ['24', '25']:
            cos.sChrID = 'chr%s' % cos.nChrID
        else:
            dict_sChrKey = {'24': 'chrX', '25': 'chrY'}
            cos.sChrID = dict_sChrKey[cos.nChrID]
        # if END

        list_sPosCheck = list(set(list_sColumn[23].split(':')[1].split('-')))

        if len(list_sPosCheck) > 1:
            cos.sPos = list_sPosCheck[0]
        else:
            cos.sPos = ''.join(list_sPosCheck)
        # if END:

        cos.sStrand = list_sColumn[24]
        cos.bSNP = True if list_sColumn[25] == 'y' else False

        altnotation = extract_mutation(cos.sAltType)

        ## Get only 1bp SNVs ##
        if altnotation == None: continue
        if len(altnotation) != 3: continue

        mutaid = int(cos.sMutaID.replace('COSM', ''))

        if mutaid not in dict_out:
            dict_out[mutaid] = ''
        dict_out[mutaid] = [cos.sGeneName, altnotation, cos.sChrID, cos.sStrand, cos.sGenicPos]

    # loop END: i, line
    inf.close()

    return dict_out

# def END: cCos_parse_cosmic_consensus

class cVCFData: pass
def parse_vcf_file(vcffile):
    if not os.path.isfile(vcffile):
        sys.exit('File Not Found %s' % vcffile)

    dict_out = {}

    if vcffile.endswith('.gz'): inf      = gzip.open(vcffile, 'rt')
    else:                       inf    = open(vcffile, 'r')

    for line in inf:

        # File Format
        # Column Number:     | 0       | 1        | 2          | 3       | 4
        # Column Description:| sChrID  | nPos     | sDBSNP_ID  | sRefNuc | sAltNuc
        # Column Example:    | 1       | 32906558 | rs79483201 | T       | A
        # Column Number:     | 5       | 6        | 7          | 8              | 9./..
        # Column Description:| fQual   | sFilter  | sInfo      | sFormat        | sSampleIDs
        # Column Example:    | 5645.6  | PASS     | .          | GT:AD:DP:GQ:PL | Scores corresponding to sFormat

        if line.startswith('#'): continue  # SKIP Information Headers
        list_col  = line.strip('\n').split('\t')

        vcf       = cVCFData()
        vcf.chrom = 'chr%s' % list_col[0]

        try:                vcf.pos = int(list_col[1])
        except ValueError:  continue

        vcf.varID   = int(list_col[2])
        vcf.ref     = list_col[3]
        vcf.alt     = list_col[4]
        vcf.qual    = float(list_col[5]) if list_col[5] != '.' else list_col[5]
        vcf.filter  = list_col[6]
        vcf.addinfo = list_col[7]

        dict_addinfo = dict([info.split('=') for info in vcf.addinfo.split(';') if len(info.split('=')) == 2])

        vcf.vartype = dict_addinfo['CLNVC'].lower()
        try: vcf.geneinfo = dict_addinfo['GENEINFO'].upper()
        except KeyError: vcf.geneinfo = 'N/A'


        if vcf.varID not in dict_out:
            dict_out[vcf.varID] = ''
        dict_out[vcf.varID] = vcf

    #loop END: sReadLine
    inf.close()

    return dict_out
#def END: parse_vcf_file


def load_clinvar_tempfile (tmpdir, id):

    id        = int(id)
    indexfile = '%s/index.txt' % tmpdir

    inf       = open(indexfile, 'r')
    for line in inf:

        idxrange, tmpfile = line.strip('\n').split('\t')

        start, end = [int(i) for i in idxrange.split('-')]

        if start <= id <= end: break
    #loop END: line
    inf.close()

    inf      = open(tmpfile, 'rb')
    dict_vcf = pickle.load(inf)
    inf.close()

    vcf = dict_vcf[id]

    return vcf
#def END: load_clinvar_tempfile


def extract_mutation(value):
    match = re.search(r'([ACGT]+)>([ACGT]+)', value)
    return f"{match.group(1)}>{match.group(2)}" if match else None


def load_cosmic_tempfile(self, tmpdir, id):

    id = int(id)
    indexfile = '%s/index.txt' % tmpdir

    inf = open(indexfile, 'r')
    for line in inf:

        idxrange, tmpfile = line.strip('\n').split('\t')

        start, end = [int(i) for i in idxrange.split('-')]

        if start <= id <= end: break
    # loop END: line
    inf.close()

    inf = open(tmpfile, 'rb')
    dict_vcf = pickle.load(inf)
    inf.close()

    vcf = dict_vcf[id]

    return vcf
# def END: load_cosmic_tempfile


# def get_positions(location):
#
#     # Extract exon positions from CDS features
#     positions = []
#     print(location)
#     for loc in location.parts:
#         start = loc.start + 1  # Add 1 to convert from 0-based to 1-based position
#         end   = loc.end
#         positions.append([start, end])
#     #loop END:
#
#     return positions

def make_geneid_prefile():
    # Lowest RNA_nucleotide_gi (column 5) indicated most tested/recommended isoform on NCBI
    # file format:
    # 0	tax_id	24
    # 1	GeneID	77267466
    # 2	status	NA
    # 3	RNA_nucleotide_accession.version	-
    # 4	RNA_nucleotide_gi	-
    # 5	protein_accession.version	WP_011787424.1
    # 6	protein_gi	500111419
    # 7	genomic_nucleotide_accession.version	NZ_CP104755.1
    # 8	genomic_nucleotide_gi	2310801391
    # 9	start_position_on_the_genomic_accession	0
    # 10	end_position_on_the_genomic_accession	1385
    # 11	orientation	+
    # 12	assembly	-
    # 13	mature_peptide_accession.version	-
    # 14	mature_peptide_gi	-
    # 15	Symbol	dnaA

    refdir = '%s/ref' % os.getcwd()

    refseq = '%s/gene2refseq.review.valid.GRCh38.tsv.gz' % refdir
    df_ref = pd.read_csv(refseq, delimiter='\t')

    gifile = '%s/nmid2gi.txt' % refdir
    dict_gi = dict(line.strip().split('\t') for line in open(gifile))

    refflat = '%s/20231023_refflat.grch38.exoncnt.txt' % refdir
    dict_exoncnt = dict(line.strip().split('\t') for line in open(refflat))

    dict_gene = {}
    for i, row in df_ref.iterrows():

        strand = row['orientation']
        nmid = row['RNA_nucleotide_accession.version']
        genesym = row['Symbol'].upper()
        exoncnt = dict_exoncnt.get(nmid, '')
        if not exoncnt: continue

        rna_gi = int(row['RNA_nucleotide_gi']) if row['RNA_nucleotide_gi'] != '-' else row['RNA_nucleotide_gi']
        genomic_gi = dict_gi.get(nmid, '')

        if genesym not in dict_gene:
            dict_gene[genesym] = []
        dict_gene[genesym].append([rna_gi, genomic_gi, exoncnt, strand])
    # loop END: i, row

    outdir = '%s/ref/rnaid' % os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    for gene, info in dict_gene.items():
        outfile = '%s/%s.txt' % (outdir, gene)
        outf = open(outfile, 'w')
        rna_gi, genomic_gi, exoncnt, strand = sorted(info, key=lambda e: e[0])[0]

        out = '%s\t%s\t%s' % (genomic_gi if genomic_gi else rna_gi, exoncnt, strand)

        outf.write(out)
        outf.close()
    # loop END: gene, list_GIs
# def END: make_geneid_prefile


def add_nmid_to_refflat():
    refdir = '%s/ref' % os.getcwd()

    refflat = '%s/gencode.v44.primary_assembly.annotation.refflat.txt' % refdir

    header = ['ensembl_gene', 'ensembl_txn', 'chrom', 'strand', 'txn_start', 'txn_end', 'cds_start', 'cds_end',
              'exon_cnt', 'exon_start', 'exon_end']
    df_ref = pd.read_csv(refflat, delimiter='\t', names=header)

    meta = '%s/gencode.v44.metadata.RefSeq.txt' % refdir
    dict_meta = dict(line.strip().split() for line in open(meta))

    header2 = ['nmid'] + header[2:]
    nmid_df = pd.DataFrame(columns=header2)

    for i, row in df_ref.iterrows():

        txnid = row['ensembl_txn']
        nmid = dict_meta.get(txnid, '')
        chrom = row['chrom']

        if not chrom.startswith('chr'): continue
        if not nmid: continue

        new_row = pd.DataFrame([[nmid] + row[2:].tolist()], columns=header2)
        nmid_df = pd.concat([nmid_df, new_row], ignore_index=True)

    # loop END: i, row
    nmid_df = nmid_df.reset_index(drop=True)
    print(nmid_df)

    outfile = '%s/gencode.v44.primary_assembly.annotation.refflat.nmid.txt' % refdir
    nmid_df.to_csv(outfile, index=False)
# def END: add_nmid_to_refflat


def make_dp_input(record, pe, flank, target):

    cds = record.cds()

    # start and end positions are index positions within the entrez source record, not actual genomic positions
    list_pos = record.get_positions(cds)
    list_seq = record.get_sequences(cds)
    targets = [i + 1 for i, pos in enumerate(list_pos)]

    if target not in targets:
        sys.exit('Exon Number is Invalid: Number of exons in CDS of %s= %s' % (record.genesym, len(list_pos)))

    # dict_out = {'ID':[], 'exon': [], 'wtseq':[], 'edseq': [], 'editpos':[], 'refnuc': [], 'altnuc':[]}
    dict_out = {}

    for i, ((s, e), seq) in enumerate(zip(list_pos, list_seq)):
        if i != target - 1: continue
        find_PAMs_for_input_v2(record, i, s, e, seq, pe, flank, dict_out)
    # loop END: i, ((s, e), seq)

    list_out = []
    for editpos, list_guidekeys in dict_out.items():

        if len(list_guidekeys) > 3:
            list_out += list_guidekeys[:3]
        else:
            list_out += list_guidekeys
    # loop END: editpos, list_guidekeys

    hder_essen = ['ID', 'target', 'wtseq', 'edseq']
    df = pd.DataFrame(list_out, columns=hder_essen)

    if len(df) == 0:
        sys.exit('No PAM sequence was detected in the exon == %s' % target)
    return df
# def END: make_dp_input


def make_dp_input_v2(fasta, pe_system, chrID, strand, coord, flank, target):

    if target == 0: # all exons
        list_targets = []

        for exon_no, poskey in coord.items():

            s, e = [int(pos) for pos in poskey.split('-')]
            seq  = fasta.fetch(chrID, s-1, e).upper()
            list_targets.append([int(exon_no), s, e, seq])
        #loop END:

        dict_out = {}
        for exon_no, s, e, seq in list_targets:
            find_PAMs_for_input_v2(exon_no, chrID, s, e, seq, fasta, pe_system, flank, dict_out)
        #loop END:

    else: #single exon
        s, e     = [int(pos) for pos in coord.split('-')]
        seq      = fasta.fetch(chrID, s - 1, e).upper()

        dict_out = {}
        find_PAMs_for_input_v2(target, chrID, s, e, seq, fasta, pe_system, flank, dict_out)

    list_out = []
    for editpos, list_guidekeys in dict_out.items():

        if len(list_guidekeys) > 3:
            list_out += list_guidekeys[:3]
        else:
            list_out += list_guidekeys
    # loop END: editpos, list_guidekeys

    hder_essen = ['ID', 'target', 'wtseq', 'edseq']
    df = pd.DataFrame(list_out, columns=hder_essen)

    if len(df) == 0:
        sys.exit('No PAM sequence was detected in the exon == %s' % target)

    #print('pegRNAs', len(df))

    return df
# def END: make_dp_input


def df_make_dp_input_clinvar (ref, fasta, inputID, target, altindex):
    dict_sCLNVCKey = {'Insertion': 'insertion',
                      'insertion': 'insertion',
                      'Duplication': 'insertion',
                      'duplication': 'insertion',

                      'Deletion': 'deletion',
                      'deletion': 'deletion',

                      'Inversion': 'substitution',
                      'single_nucleotide_variant': 'substitution',
                      'single nucleotide variant': 'substitution'}

    if inputID.startswith('VCV'):
        query = int(inputID.split('.')[0].replace('VCV', ''))
    else:
        query = int(inputID)

    tmpdir   = '%s/clinvar_temp' % ref
    vcf      = load_index_tempfile(tmpdir, query)
    genesym  = vcf.geneinfo.split(':')[0].upper()
    vartype  = vcf.vartype
    seq      = fasta.fetch(vcf.chrom, vcf.pos - altindex - 1, vcf.pos + altindex).upper()

    editinfo = determine_alttype_altlen(vartype, vcf.ref, vcf.alt, dict_sCLNVCKey)

    try:
        dict_refAA = genet.utils.RefAA(genesym).dict_refAA
        refAA = dict_refAA[vcf.pos]
    except:
        editinfo = 'Invalid ClinVar ID, ClinVar GeneInfo Not found'


    if editinfo.startswith('Invalid'):
        return [], editinfo, {}

    elif editinfo != 'sub1':
        message = 'Invalid Edit Type:\\nFound: %s' % editinfo
        return [], message, {}

    else:
        if target == 'model':
            wtseq = seq
            edseq = seq[:altindex] + vcf.alt + seq[altindex + len(vcf.ref):]

        else:
            wtseq = seq[:altindex] + vcf.alt + seq[altindex + len(vcf.ref):]
            edseq = seq
        # if END:

        id       = '%s|%s:%s|%s>%s' % (query, vcf.chrom, vcf.pos, vcf.ref, vcf.alt)
        list_out = [[id, target, wtseq, edseq]]

        header = ['ID', 'target', 'wtseq', 'edseq']
        df    = pd.DataFrame(list_out, columns=header)
        return df, editinfo, dict_refAA

    # cv_record   = db.GetClinVar(inputID)
    # wt, ed      = cv_record.seq()
    # edittype    = cv_record.alt_type
    # editsize    = int(cv_record.alt_len)
    # start       = cv_record.start
    # end         = cv_record.stop
    # refnuc      = cv_record.ref_nt
    # altnuc      = cv_record.alt_nt

    # if target == 'model':
    #     wtseq = wt
    #     edseq = ed
    #
    # else:
    #     wtseq = ed
    #     edseq = wt
#def END: df_make_dp_input_clinvar


def df_make_dp_input_cosmic (ref, fasta, inputID, target, altindex):

    if inputID.startswith('COSM'):
        query = int(inputID.split('.')[0].replace('COSM', ''))
    else:
        query = int(inputID)

    tmpdir      = '%s/cosmic_temp' % ref

    try: data        = load_index_tempfile(tmpdir, query)
    except KeyError:
        messsage = 'Invalid COSMIC ID, Please Check the COSMIC ID'
        return [], messsage

    genesym     = data[0].split('_')[0].upper()
    altnotation = data[1]
    ref, alt    = altnotation.split('>')
    chrom       = data[2]
    strand      = data[3]
    coords      = data[4]
    s, e        = [int(pos) for pos in coords.split(':')[-1].split('-')]
    seq         = fasta.fetch(chrom, s - altindex - 1, e + altindex, strand).upper()
    editinfo    = 'sub1'

    try:
        dict_refAA = genet.utils.RefAA(genesym).dict_refAA
        refAA = dict_refAA[s]
    except:
        editinfo = 'Invalid COSMIC ID, COSMIC Gene Not found'

    if editinfo.startswith('Invalid'):
        return [], editinfo, {}
    else:
        if target == 'model':
            wtseq = seq
            edseq = seq[:altindex] + alt + seq[altindex + len(ref):]

        else:
            wtseq = seq[:altindex] + alt + seq[altindex + len(alt):]
            edseq = seq
        # if END:

        id       = '%s|%s:%s|%s>%s' % (query, chrom, s, ref, alt)
        list_out = [[id, target, wtseq, edseq]]

        header = ['ID', 'target', 'wtseq', 'edseq']
        df    = pd.DataFrame(list_out, columns=header)
        return df, editinfo, dict_refAA
#def END: df_make_dp_input_cosmic


def df_make_dp_input_seq (wtseq, edseq, target, altindex):

    ref         = wtseq[altindex]
    alt         = edseq[altindex]

    id          = '%s|-|%s>%s' % ('seq_query', ref, alt)
    list_out    = [[id, target, wtseq, edseq]]

    header      = ['ID', 'target', 'wtseq', 'edseq']
    df          = pd.DataFrame(list_out, columns=header)
    return df

#def END: df_make_dp_input_seq


def df_make_dp_input_pos (fasta, chrom, coords, target, altindex):

    if '-' in coords:
        s, e     = [int(pos) for pos in coords.split('-')]
    else: s, e = int(coords), int(coords)

    list_out = []
    for i, pos in enumerate(range(s, e+1)):
        query   = 'pos_query_%s' % i
        seq      = fasta.fetch(chrom, pos-1 - altindex, pos + altindex).upper()
        ref      = seq[altindex]

        for nuc in 'ACGT':

            if ref == nuc: continue  # Skip same base as WT

            wtseq    = seq
            edseq    = seq[:altindex] + nuc + seq[altindex + len(ref):]

            id       = '%s|%s:%s|%s>%s' % (query, chrom, pos, ref, nuc)
            list_out = [[id, target, wtseq, edseq]]
    #loop END:
    header      = ['ID', 'target', 'wtseq', 'edseq']
    df          = pd.DataFrame(list_out, columns=header)
    return df
#def END: df_make_dp_input_seq



def load_index_tempfile (tmpdir, id):

    id        = int(id)
    indexfile = '%s/index.txt' % tmpdir

    inf       = open(indexfile, 'r')
    for line in inf:

        idxrange, tmpfile = line.strip('\n').split('\t')

        start, end = [int(i) for i in idxrange.split('-')]

        if start <= id <= end: break
    #loop END: line
    inf.close()

    inf      = open(tmpfile, 'rb')
    dict_data = pickle.load(inf)
    inf.close()

    data = dict_data[id]

    return data
#def END: load_clinvar_tempfile


def determine_alttype_altlen (vartype, ref_nt, alt_nt, dict_sCLNVCKey):

    if vartype in ['Microsatellite', 'Indel']:
        if len(ref_nt) > len(alt_nt):
            alttype = 'deletion'
            altlen  = len(ref_nt) - 1

            if ref_nt[0] != alt_nt[0]:
                return 'Invalid Ref-Alt nucleotides:\\nFound: %s -> %s' % (ref_nt, alt_nt) # ex) GAGA -> TCT

        elif len(ref_nt) < len(alt_nt):
            alttype = 'insertion'
            altlen  = len(alt_nt) - 1

            if ref_nt[0] != alt_nt[0]:
                return 'Invalid Ref-Alt nucleotides:\\nFound: %s -> %s' % (ref_nt, alt_nt) # ex) GAGA -> TCT
        else:
            alttype = 'substitution'
            altlen  = len(alt_nt)
    else:
        try: alttype = dict_sCLNVCKey[vartype]
        except KeyError: return 'Invalid Variant Type:\\nFound: %s' % (vartype)

        if alttype == 'deletion':
            altlen = len(ref_nt) - 1

        elif alttype == 'insertion':
            altlen = len(alt_nt) - 1
        else:
            altlen = len(alt_nt)

    if altlen > 3: return 'Invalid Variant Size:\\nFound: %snt' % (altlen)

    editinfo = '%s%s' % (alttype[:3], altlen)

    return editinfo
#def END: determine_alttype_altlen


def find_PAMs_for_input(record, exon_no, exon_start, exon_end, seq, pe_system, flank, dict_out):
    ## Parameters ##
    guidelen   = 20
    max_pbslen = 17
    max_rttlen = 40
    if 'NRCH' in pe_system:  # for NRCH-PE PAM
        dict_pam_re = {'+': '[ACGT][ACGT]G[ACGT]|[ACGT][CG]A[ACGT]|[ACGT][AG]CC|[ATCG]ATG',
                       '-': '[ACGT]C[ACGT][ACGT]|[ACGT]T[CG][ACGT]|G[GT]T[ACGT]|ATT[ACGT]|CAT[ACGT]|GGC[ACGT]|GTA[ACGT]'}
    else:
        dict_pam_re = {'+': '[ACGT]GG', '-': 'CC[ACGT]'}  # for Original-PE NGG PAM

    # if record.strand == '-':
    #     fullseq = revcom(record.fullseq)
    #     exonseq = revcom(seq)

    for strand in ['+']:
        pam_re = dict_pam_re[strand]

        for match in regex.finditer(pam_re, seq, overlapped=True):

            i_start = match.start()
            i_end = match.end()
            pamseq = seq[i_start:i_end]

            if exon_no == 0 and i_start in [0, 1, 2]: continue  # SKIP PAM overlapping with first and last codon

            if strand == '+':
                g_start = exon_start + i_start - guidelen + 1
                g_end = exon_start + i_end
                nickpos = exon_start + i_start - 3
                guideseq = record.fullseq[g_start:g_end]

                winsize = max_rttlen if (exon_end - nickpos) > max_rttlen else exon_end - nickpos
                alt_window = record.fullseq[nickpos:(nickpos + winsize)]
                edit_start = nickpos

            else:
                g_start = exon_start + i_start
                g_end = exon_start + i_end + guidelen - 1
                nickpos = exon_start + i_end + 3
                guideseq = revcom(record.fullseq[g_start:g_end])
                winsize = max_rttlen if (nickpos - exon_start) > max_rttlen else nickpos - exon_start
                alt_window = revcom(record.fullseq[(nickpos - winsize):nickpos])
                edit_start = nickpos - winsize
            # if END:

            # if guideseq == 'CCCTTCTCAGGATTCCTACAGG':
            #
            #     print(strand)
            #     print(seq)

            #     print(guideseq, pamseq)
            #     print(alt_window, winsize)
            #

            # print(exon_start, exon_end)
            # print(g_start, g_end)
            # print(guideseq, pamseq)
            # print(alt_window, winsize)
            # #
            # # print(revcom(seq))
            # # print(revcom(guideseq), revcom(pamseq))
            # # print(revcom(alt_window), winsize)
            # # print(revcom(record.fullseq[edit_start-2:edit_start+2]))
            # #
            # sys.exit()

            if not exon_start <= edit_start <= exon_end: continue

            loc_start, loc_end = check_genic_locale(record, exon_start, exon_end, g_start, g_end, strand)
            inputseqs = get_all_sub_combo(record.fullseq, alt_window, edit_start, flank, strand)

            for i, (wt, ed, editpos, refnuc, altnuc) in enumerate(inputseqs):

                if record.strand == '-':
                    corrected_strand = '-' if strand == '+' else '+'
                else:
                    corrected_strand = strand

                guidekey = '%s,%s,%s,%s:%s,%s>%s' % (
                guideseq, corrected_strand, editpos, loc_start, loc_end, refnuc, altnuc)
                #
                print(wt, refnuc)
                print(ed, altnuc)

                if editpos not in dict_out:
                    dict_out[editpos] = []
                dict_out[editpos].append([guidekey, 'exon%s' % (exon_no + 1), wt, ed])
            # loop END:
        # loop END: match
    # loop END: sStrand
# def END: find_PAMs_for_input


def get_all_sub_combo(fullseq, targetseq, edit_start, flank, strand):

    inputseqs = []
    for i in range(len(targetseq)):
        for alt in 'ACGT':
            if targetseq[i] == alt: continue  # Skip same base as WT

            wtseq = fullseq[edit_start - flank + i:edit_start + i] + targetseq[i] + fullseq[
                                                                                    edit_start + i + 1:edit_start + i + 1 + flank]
            edseq = fullseq[edit_start - flank + i:edit_start + i] + alt + fullseq[
                                                                           edit_start + i + 1:edit_start + i + 1 + flank]

            if strand == '-':
                wtseq  = revcom(wtseq)
                edseq  = revcom(edseq)
                refnuc = revcom(targetseq[i])
                altnuc = revcom(alt)
                inputseqs.append([wtseq, edseq, edit_start + i, refnuc, altnuc])

            else:
                inputseqs.append([wtseq, edseq, edit_start + i, targetseq[i], alt])

        # loop END:
    # loop END:

    return inputseqs
#def END: get_all_sub_combo


def get_all_sub_combo_v2(chrID, fasta, targetseq, altwin, flank, strand):

    inputseqs = []

    for i, pos in enumerate(range(altwin[0], altwin[1])):

        start = pos - flank - 1
        end   = pos + flank

        wtseq = fasta.fetch(chrID, start, end)

        for alt in 'ACGT':
            if targetseq[i] == alt:
                continue  # Skip same base as WT

            edseq = wtseq[:flank] + alt + wtseq[flank+1:]

            if strand == '-':
                wt     = revcom(wtseq)
                ed     = revcom(edseq)
                refnuc = revcom(targetseq[i])
                altnuc = revcom(alt)
                inputseqs.append([wt, ed, pos, refnuc, altnuc])

            else:
                inputseqs.append([wtseq, edseq, pos, targetseq[i], alt])

        # loop END:
    # loop END:

    return inputseqs
#def END: get_all_sub_combo


def find_PAMs_for_input_v2(exon_no, chrID, exon_start, exon_end, seq, fasta, pe_system, flank, dict_out):

    ## Parameters ##
    max_rttlen = 40
    guidelen   = 20
    if 'NRCH' in pe_system:  # for NRCH-PE PAM
        dict_pam_re = {'+': '[ACGT][ACGT]G[ACGT]|[ACGT][CG]A[ACGT]|[ACGT][AG]CC|[ATCG]ATG',
                       '-': '[ACGT]C[ACGT][ACGT]|[ACGT]T[CG][ACGT]|G[GT]T[ACGT]|ATT[ACGT]|CAT[ACGT]|GGC[ACGT]|GTA[ACGT]'}
    else:
        dict_pam_re = {'+': '[ACGT]GG', '-': 'CC[ACGT]'}  # for Original-PE NGG PAM

    for strand in ['+', '-']:
        pam_re = dict_pam_re[strand]

        for match in regex.finditer(pam_re, seq, overlapped=True):

            i_start = match.start()
            i_end   = match.end()

            if strand == '+':
                pam        = seq[i_start:i_end]
                nickpos    = i_start - 3
                winsize    = nickpos + max_rttlen

                if nickpos < 0: nickpos = 0
                if winsize > len(seq): winsize = len(seq)

                alt_window = [exon_start + nickpos, exon_start + winsize]

            else:
                pam         = seq[i_start:i_end]
                nickpos     = i_end + 3
                winsize     = nickpos - max_rttlen

                if nickpos > len(seq): nickpos = len(seq)
                if winsize < 0: winsize = 0

                alt_window = [exon_start + winsize, exon_start + nickpos]

            # if END:

            targetseq = fasta.fetch(chrID, alt_window[0]-1, alt_window[1]-1).upper()
            #loc_start, loc_end = check_genic_locale(record, exon_start, exon_end, g_start, g_end, strand)
            inputseqs = get_all_sub_combo_v2(chrID, fasta, targetseq, alt_window, flank, strand)

            for i, (wt, ed, editpos, refnuc, altnuc) in enumerate(inputseqs):

                guidekey = '%s.%s|%s:%s|%s>%s' % (exon_no, editpos, chrID, editpos, refnuc, altnuc)

                #print(guidekey)
                # print(wt)
                # print(ed)

                if editpos not in dict_out:
                    dict_out[editpos] = []
                dict_out[editpos].append([guidekey, exon_no, wt, ed])
            # loop END:

        # loop END: match
    # loop END: sStrand
    return dict_out

# def END: find_PAMs_for_input_v2


def check_genic_locale(record, exon_start, exon_end, g_start, g_end, strand):
    cds_start = record.cds().location.start
    cds_end = record.cds().location.end
    loc_start = ''
    loc_end = ''

    if strand == '+':
        if g_start <= cds_start:
            loc_start = '5utr'
        elif g_start >= cds_end:
            loc_start = '3utr'
        else:
            if exon_start <= g_start <= exon_end:
                loc_start = 'exon'
            else:
                loc_start = 'intron'
            # if END:
        # if END:

        if g_end <= cds_start:
            loc_end = '5utr'
        elif g_end >= cds_end:
            loc_end = '3utr'
        else:
            if exon_start <= g_end <= exon_end:
                loc_end = 'exon'
            else:
                loc_end = 'intron'
            # if END:
        # if END:

    else:  # -strand
        if g_start <= cds_start:
            loc_start = '3utr'
        elif g_start >= cds_end:
            loc_start = '5utr'
        else:
            if exon_start <= g_start <= exon_end:
                loc_start = 'exon'
            else:
                loc_start = 'intron'
            # if END:
        # if END:

        if g_end <= cds_start:
            loc_end = '3utr'
        elif g_end >= cds_end:
            loc_end = '5utr'
        else:
            if exon_start <= g_end <= exon_end:
                loc_end = 'exon'
            else:
                loc_end = 'intron'
            # if END:
        # if END:
    # if END:
    return [loc_start, loc_end]
# def END: check_genic_locale


def mp_run_dp(tempdir, model_dir, results, df, dict_opts):

    pe               = dict_opts['pe']
    celltype         = dict_opts['celltype']
    target           = dict_opts['target']
    top              = dict_opts['top']
    intron           = dict_opts['intron']
    inputtype        = dict_opts['inputtype']

    list_index       = df.index

    if inputtype in ['GeneSym', 'NMID', 'EnsemblID', 'HGNC', 'Position']:
        bin_cnt    = 3
    else:
        bin_cnt    = 1

    if target == 0:
        target_dir = '%s/full_run' % tempdir

    else:
        target_dir = '%s/%s' % (tempdir, target)
        os.makedirs(target_dir, exist_ok=True)

    bins = np.array_split(list_index, bin_cnt)

    list_parameters = []
    for i, bin in enumerate(bins):
        gpu = i % NUM_GPUs  # get gpu index
        list_parameters.append([i, target_dir, df, bin, gpu, dict_opts])
    # loop END: i, bin

    mp.set_start_method('spawn')
    p = mp.Pool(bin_cnt)
    p.map_async(run_dp, list_parameters).get()
    p.close()
    p.join()

    ## Combined and Sort ##
    full_dfs = []
    dfs      = []

    targetfiles = [dir for dir in os.listdir(target_dir)]

    if not targetfiles: # Check no PAM / pegRNAs #
        sys.exit('No PAM / pegRNAs Found, Try PE variants with expanded PAM repertoire')

    for exon_no in targetfiles:

        if target == 0:
            datafiles = os.listdir('%s/%s' % (target_dir, exon_no))
            for data in datafiles:
                pkl = open('%s/%s/%s' % (target_dir, exon_no, data), 'rb')
                df = pickle.load(pkl)
                pkl.close()
                dfs.append(df)
                full_dfs.append(df)
            #loop END: data
        else:

            pkl = open('%s/%s' % (target_dir, exon_no), 'rb')
            df = pickle.load(pkl)
            pkl.close()
            dfs.append(df)
            full_dfs.append(df)
    # loop END: exon_no

    # Outputs top exon tarets and total exon dp scores for normalization
    df_tar         = pd.concat(dfs, ignore_index=True).sort_values(by='%s_score' % pe, ascending=False)
    df_scores_exon = pd.DataFrame(df_tar[f'{pe}_score'])
    df_scores_exon.to_csv('%s_target_zscores.csv' % target_dir)
    df_tar.to_csv('%s.csv' % target_dir)

    # if intron == 'F':
    #     df_tar = df_tar[~df_tar['ID'].str.contains('intron')]
    #     if len(df_tar) == 0:
    #         sys.exit('All pegRNAs filtered out by Parameter: Intron Overhang= %s' % intron)

    if inputtype in ['GeneSym', 'NMID', 'EnsemblID', 'HGNC']:
        # Calculates DPscores and Zscore for entire gene (if full gene run) else same as per sexon
        dpscores  = pd.concat(full_dfs, ignore_index=True).sort_values(by='%s_score' % pe, ascending=False)[f'{pe}_score']
        mean      = dpscores.mean()
        std       = dpscores.std()
        zscores   = [(score - mean) / std for score in dpscores]

        outf = open('%s_mean_std.csv' % target_dir, 'w')
        outf.write('%s,%s' % (mean, std))
        outf.close()
        df_scores = pd.DataFrame({f'{pe}_score': dpscores, 'Zscores': zscores})


    elif inputtype in ['ClinVar', 'COSMIC', 'Sequence', 'Position']:
        dpmean     = float(open('%s/dp_mean.csv' % model_dir, 'r').readline())  # DP scores from test set
        dpstd      = float(open('%s/dp_std.csv' % model_dir, 'r').readline())   # DP scores from test set
        dpscores   = pd.concat(full_dfs, ignore_index=True).sort_values(by='%s_score' % pe, ascending=False)[f'{pe}_score']
        zscores    = [(score - dpmean) / dpstd for score in dpscores]

        df_scores = pd.DataFrame({f'{pe}_score': dpscores, 'Zscores': zscores})
    #if END:

    df_scores.to_csv('%s/%s_zscores.csv'  % (tempdir, target), index=False)

    if target == 0:

        os.system('cp %s.csv %s/%s.csv' % (target_dir, results, target))


    else:
        os.system('cp %s/%s.csv %s/%s.csv'    % (tempdir, target, results, target))
# def END: mp_run_dp


def run_dp(parameters):

    i, target_dir, df, list_index, gpu, dict_opts = parameters

    pe        = dict_opts['pe']
    celltype  = dict_opts['celltype']

    edittype  = dict_opts['edittype']
    editsize  = dict_opts['editsize']

    rtt_max   = dict_opts['rtt_max']
    pbs_max   = dict_opts['pbs_max']
    pbs_min   = dict_opts['pbs_min']
    inputtype = dict_opts['inputtype']
    target    = dict_opts['target']

    for index in list_index:
        row = df.iloc[index]
        ID  = row['ID']
        wt  = row['wtseq']
        ed  = row['edseq']
        pam = 'NRCH' if pe.startswith('NRCH') else 'NGG'

        if inputtype in ['GeneSym', 'NMID', 'EnsemblID', 'HGNC'] and target == 0:
            exon   = ID.split('.')[0]
            outdir = '%s/%s' % (target_dir, exon)
            os.makedirs(outdir, exist_ok=True)

            csv  = '%s/%s.csv.pkl' % (outdir, index)
        else:
            csv  = '%s/%s.csv.pkl' % (target_dir, index)

        dp  = prd.DeepPrime(sID=ID, Ref_seq=wt, ED_seq=ed, edit_type=edittype, edit_len=int(editsize),
                           pam=pam, pbs_min=pbs_min, pbs_max=pbs_max,
                           rtt_min=0, rtt_max=rtt_max, silence=True,
                           gpu=int(gpu))
        if dp.pegRNAcnt == 0:
            continue

        df_dpscores        = dp.predict(pe_system=pe, cell_type=celltype, show_features='syn_pe')
        sorted_df          = df_dpscores.sort_values(by='%s_score' % pe, ascending=False)
        sorted_df['wtseq'] = wt

        pkl = open(csv, 'wb')
        pickle.dump(sorted_df, pkl)
        pkl.close()
    # loop END:
# def END: run_dp


def snv_marker_old(record, df, pe_system):
    totalpegRNAs = 0

    print(len(df))

    for index, row in df.iterrows():

        ID = row['ID']
        exon = row['exon']
        wt = row['wtseq']
        ed = row['edseq']

        workdir = '%s/temp/%s/%s/%s' % (os.getcwd(), record.genesym, pe_system, exon)
        os.makedirs(workdir, exist_ok=True)

        dp = prd.DeepPrime(ID, wt, ed, 'sub', 1, silence=True, gpu=0)
        if dp.pegRNAcnt == 0: continue

        totalpegRNAs += dp.pegRNAcnt

        df_dpscores = dp.predict('PE2max', show_features=True)
        sorted_df = df_dpscores.sort_values(by='PE2max_score', ascending=False)
        # sorted_df.to_csv ('%s/%s.csv' % (workdir, index))
        dp_record = sorted_df.iloc[0]
        print(dp_record)
        sys.exit()

        print(ID)

        synony_pegrna = design.SynonymousPE(dp_record, ref_seq=wt, frame=0)
        pegrna_ext = synony_pegrna.extension

        print(dp_record.WT74_On)
        print(dp_record.Edited74_On)
        print(dp_record.Edited74_On.replace('x', ''))
        print(pegrna_ext)

        for a, b, in synony_pegrna.dict_mut.items():
            print(a, b)

        sys.exit()
    # loop END:

    print('totalpegRNAs', totalpegRNAs)


# def END: run_dp


def mp_snv_marker(analysistag, tempdir, model_dir, results, dict_opts, dict_refAA):

    pe               = dict_opts['pe']
    celltype         = dict_opts['celltype']
    target           = dict_opts['target']
    top              = dict_opts['top']

    intron           = dict_opts['intron']
    marker           = dict_opts['marker']

    base             = dict_opts['base']
    inputtype        = dict_opts['inputtype']

    if inputtype in ['GeneSym', 'NMID', 'EnsemblID', 'HGNC']:
        infile       = '%s/%s.csv' % (results, target)
        df           = pd.read_csv(infile)
        df           = df.drop(df.columns[0], axis=1)
        if target == 0:
            mean, std    = list(map(float, open('%s/full_run_mean_std.csv' % (tempdir), 'r').readline().split(',')))
        else:
            mean, std    = list(map(float, open('%s/%s_mean_std.csv' % (tempdir, target), 'r').readline().split(',')))
        #df_scores    = pd.read_csv('%s/exon%s_zscores.csv'           % (tempdir, target))
        df_scores   = pd.read_csv('%s/dpscores.csv' % model_dir, names=['%s_score' % pe, 'Zscores'])

    elif inputtype in ['ClinVar', 'COSMIC', 'Sequence', 'Position']:
        infile      = '%s/%s.csv' % (results, target)
        df          = pd.read_csv(infile)
        df          = df.drop(df.columns[0], axis=1)
        df_scores   = pd.read_csv('%s/dpscores.csv' % model_dir, names=['%s_score' % pe, 'Zscores'])  # DP scores from test set
        mean        = float(open('%s/dp_mean.csv' % model_dir, 'r').readline())
        std         = float(open('%s/dp_std.csv' % model_dir, 'r').readline())
    #if END:

    plots        = '%s/matplot' % results
    plotphp      = '/SynDesign/data/%s/results/matplot' % analysistag #formatted for php
    os.makedirs(plots, exist_ok=True)

    list_index   = df.index
    bin_cnt      = 24 #CPU run
    bins         = np.array_split(list_index, bin_cnt)

    results_temp    = '%s/temp' % results
    os.makedirs(results_temp, exist_ok=True)

    list_column_simp = ['id', 'GuideSeq', 'PBSlen', 'RTlen', 'RHA_len', '%s_score' % pe,
                       'WT74_On', 'WT74_Strand', 'Edited74_On', 'wtseq',
                       'Edit_pos', 'Mut_pos',
                       'Codon_WT', 'Codon_Mut', 'RTT_DNA_frame', 'RTT_DNA_Strand',
                       'PAM_Mut', 'Edit_class',
                       'pbsrtt', 'pbsrtt_wSyn', 'genic', 'altnote', 'altnote_AA',
                       'spacer_top', 'spacer_bot', 'ext_top', 'ext_bot', 'ext_top_wSyn', 'ext_bot_wSyn',
                       'Edited_wNote', 'Edited_wNoteSyn', 'Zscore', 'percentile', 'plotpath']

    list_binfiles   = []
    list_parameters = []
    for i, bin in enumerate(bins):
        bin_s, bin_e = bin[0], bin[-1]
        outfile      = '%s/%s_%s-%s.csv' % (results_temp, target, bin_s, bin_e)
        list_binfiles.append(outfile)
        list_parameters.append([i, results_temp, df, bin_s, bin_e, mean, std, plotphp, df_scores, list_column_simp, outfile, dict_opts, dict_refAA])
    # loop END: i, bin

    p2 = mp.Pool(bin_cnt)
    p2.map_async(run_snv_marker, list_parameters).get()
    p2.close()
    p2.join()

    df_bins = []
    for binfile in list_binfiles:
        df_bin   = pd.read_csv(binfile)
        df_bins.append(df_bin)
    # loop END: exon_no

    list_forHTML = ['id', 'genic', 'altnote', 'altnote_AA', 'GuideSeq', 'pbsrtt', 'pbsrtt_wSyn', '%s_score' % pe, 'Zscore', 'percentile',

                    'PBSlen', 'RTlen', 'RHA_len',

                    'WT74_On', 'WT74_Strand', 'Edited74_On', 'Edited_wNote', 'Edited_wNoteSyn',

                    'Edit_pos', 'Edit_class',

                    'spacer_top', 'spacer_bot', 'ext_top', 'ext_bot',  'ext_top_wSyn', 'ext_bot_wSyn',

                    'plotpath']

    df_full  = pd.concat(df_bins, ignore_index=True).sort_values(by='%s_score' % pe, ascending=False)

    outfile = '%s/%s_fullout.csv' % (results, base)
    df_full.to_csv(outfile, index=False)

    df_html  = df_full[list_forHTML].head(top)
    outfile = '%s/%s_html.csv' % (results, base)
    df_html.to_csv(outfile, index=False)

    for i, row in df_html.iterrows():

        outplot = '%s/%s.png' % (plots, row['id'])
        zscore  = row['Zscore']
        rank_plots(outplot, zscore, df_scores['Zscores'], pe, celltype)
    #loop END:
    os.system('rm -rf %s' % results_temp)
    os.system('rm -rf %s' % infile)

#def END: mp_snv_marker


def run_snv_marker (parameters):
    i, results_temp, df, s, e, mean, std, plotphp, df_scores, list_col_simp, outfile, dict_opts, dict_refAA = parameters

    pe        = dict_opts['pe']
    target    = dict_opts['target']
    intron    = dict_opts['intron']
    marker    = dict_opts['marker']
    inputtype = dict_opts['inputtype']

    df_output    = []
    for index, dp_record in df.loc[s:e].iterrows():

        guideID, genic, altnote     = dp_record['ID'].split('|')

        chrID, pos = genic.split(':') if genic != '-' else 'NA'
        ref, alt   = altnote.split('>')

        if inputtype in ['Sequence', 'Position']: #currently AA and framing not availble for Sequence inputs, need to add frame input for sequence and position
            chrID, pos  = '', ''
            codon       = ''
            frame       = ''
            strand      = ''
            aa          = ''
            aa_index    = ''
            mut_codon   = ''
            altaa       = ''
            altnote_AA  = '-'
            frame       = 0
        else:
            chrID, pos = genic.split(':')
            codon     = (dict_refAA[int(pos)]['codon'])
            frame     = (dict_refAA[int(pos)]['frame'])
            strand    = (dict_refAA[int(pos)]['strand'])
            aa        = (dict_refAA[int(pos)]['aa'])
            aa_index  = (dict_refAA[int(pos)]['codon_i'])

            if strand == '+':
                mut_codon = codon[:frame] + alt + codon[frame+1:]
            else:
                mut_codon = codon[:frame] + revcom(alt) + codon[frame + 1:]
                #mut_codon = revcom(mut_codon)

            altaa       = genet.utils.translate(mut_codon)
            altnote_AA  = '%s%s%s' % (aa, aa_index, altaa)

        syn_pe              = design.SynonymousPE(dp_record, ref_seq=dp_record['wtseq'], frame=frame)

        if syn_pe.output is None: continue #No syn marker available

        df_out                      = syn_pe.output
        df_out['pbsrtt']            = syn_pe.pbs_dna + syn_pe.rtt_dna
        df_out['pbsrtt_wSyn']       = syn_pe.extension

        syn_locale                  = df_out['Edit_class'].split('_')[0]
        if marker != 'any':
            if marker != syn_locale: continue

        dp_combine                  = pd.concat([dp_record, df_out], axis=0)

        gN19                        = 'G' + dp_record['GuideSeq'][1:-3]
        idnumber                    = '%s_%s' % (guideID, index)

        dp_combine['id']            = idnumber
        dp_combine['genic']         = genic
        dp_combine['altnote']       = altnote.replace('>', '/')
        dp_combine['altnote_AA']    = altnote_AA
        editpos = int(dp_combine['Edit_pos'])
        mutpos  = int(dp_combine['Mut_pos'])
        pbslen  = int(dp_combine['PBSlen'])
        rtlen   = int(dp_combine['RTlen'])

        ## Check MutPos Locale
        if strand == '+':
            genic_mutpos = (int(pos) - editpos) + mutpos
        else:
            genic_mutpos = (int(pos) - (rtlen - editpos)) + (rtlen - mutpos)

        try:
            dict_mutpos = dict_refAA[genic_mutpos]
        except KeyError: continue  #MutPos not in CDS

        dp_combine['WT74_Strand']   = '%s (%s)' % (dp_combine['WT74_On'], dp_combine['GuideStrand'])

        dp_combine['spacer_top']    = 'cacc%sgtttt' % gN19
        dp_combine['spacer_bot']    = 'ctctaaaac%s' % revcom(gN19)
        dp_combine['ext_top']       = 'gtgc%s' % revcom(df_out['pbsrtt'])
        dp_combine['ext_bot']       = 'aaaa%s' % df_out['pbsrtt']
        dp_combine['ext_top_wSyn']  = 'gtgc%s' % revcom(df_out['pbsrtt_wSyn'])
        dp_combine['ext_bot_wSyn']  = 'aaaa%s' % df_out['pbsrtt_wSyn']

        buffer5 = 21 - pbslen
        buffer3 = 53 - rtlen

        dp_combine['Edited_wNote']    = ' ' * buffer5 + get_ext_edited_notation(dp_combine['WT74_On'],
                                                                             dp_combine['Edited74_On'])
        dp_combine['Edited_wNoteSyn'] = ' ' * buffer5 + get_ext_edited_notation(dp_combine['WT74_On'][buffer5:-buffer3],
                                                                                dp_combine['pbsrtt_wSyn'])

        dp_combine['Zscore']     = (dp_combine[f'{pe}_score'] - mean) / std
        dp_combine['percentile'] = '%sth' % (int(stats.percentileofscore(df_scores['Zscores'], dp_combine['Zscore'])))

        dp_combine['plotpath']   = '%s/%s.png' % (plotphp, idnumber)

        df_output.append(dp_combine)
    # loop END:
    list_column = ['ID', 'WT74_On', 'Edited74_On', 'GuideSeq', 'GuideStrand', 'PBSlen', 'RTlen',
                   'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', '%s_score' % pe,  'wtseq',
                   'Codon_WT', 'Codon_Mut', 'RTT_DNA_frame', 'RTT_DNA_Strand',
                   'AminoAcid_WT', 'AminoAcid_Mut', 'Silent_check', 'Mut_pos',
                   'Mut_refpos', 'PAM_Mut', 'Priority', 'Edit_class', 'RTT_DNA_Mut',
                   'pbsrtt', 'pbsrtt_wSyn',

                   'id', 'genic', 'altnote', 'altnote_AA', 'WT74_Strand',

                   'spacer_top', 'spacer_bot', 'ext_top', 'ext_bot', 'ext_top_wSyn', 'ext_bot_wSyn',

                   'Edited_wNote', 'Edited_wNoteSyn', 'Zscore', 'percentile', 'plotpath']
    df_out  = pd.DataFrame(df_output, columns=list_column)

    df_simp = df_out[list_col_simp]
    df_simp.to_csv(outfile, index=False)
# def END: run_snv_marker

def get_ext_edited_notation(ref, alt):
    seq = ""

    for ref_nuc, alt_nuc in zip(ref, alt):

        if 'x' in [ref_nuc, alt_nuc]: continue

        if ref_nuc == alt_nuc:
            seq += ref_nuc
        else:
            seq += f"({ref_nuc}/{alt_nuc})"
    # loop END:

    return seq
# def END: get_ext_edited_notation



def rank_plots (outplot,  zscore, list_zscores, pe, celltype):

    label = '%s-%s' % (pe, celltype)
    list_minmax = minmax_scale(list_zscores)
    matplot_scatter (label, list_zscores, list_minmax, outplot, zscore)
# def END: rank_plots


def matplot_scatter(pe, list_scores, list_percents, outf, rank):
    list_X, list_Y  = [], []
    for x, y in zip(list_scores, list_percents):
        list_X.append(x)
        list_Y.append(y*100)

    assert len(list_X) == len(list_Y)
    ### Figure Size ###
    FigWidth         = 12
    FigHeight        = 5

    OutFig  = plt.figure(figsize=(FigWidth, FigHeight))
    SubPlot = OutFig.add_subplot(111)

    ### Marker ###########
    Red             = 0
    Green           = 0
    Blue            = 0
    MarkerSize      = 25
    Circle          = 'o'
    DownTriangle    = 'v'
    #######################

    ### Log Start Point ###
    LimitThresh     = 10
    #######################

    ### Axis Range ########
    Xmin = min(list_scores)
    Xmax = max(list_scores)
    Ymin = 0
    Ymax = 100
    ########################

    ### Tick Marks #########
    TickScale     = 1
    MajorTickSize = 10
    MinorTickSize = 5

    plt.xlim(xmin=Xmin, xmax=Xmax)
    plt.ylim(ymin=Ymin, ymax=Ymax)
    #plt.xscale('symlog', linthreshx=LimitThresh)
    #plt.yscale('symlog', linthreshy=LimitThresh)

    #plt.axes().xaxis.set_minor_locator(MinorSymLogLocator(TickScale))
    #plt.axes().yaxis.set_minor_locator(MinorSymLogLocator(TickScale))

    plt.tick_params(which='major', length=MajorTickSize)
    plt.tick_params(which='minor', length=MinorTickSize)

    #Set xticks as percentiles
    #p = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    #plt.yticks((len(list_Y) - 1) * p / 100., map(str, p))

    plt.title('Comparison with Z-scores from test data: %s' % pe)
    plt.vlines(rank, ymin=Ymin, ymax=Ymax, colors='red')
    plt.xlabel('Z-Score')
    plt.ylabel('Score Rank')
    SubPlot.scatter(list_X, list_Y, marker=Circle, color='black', s=MarkerSize)
    OutFig.savefig(outf)
#def END: matplot


def sp_run_dp(record, df, pe_system):
    totalpegRNAs = 0

    for index, row in df.iterrows():

        ID = row['ID']
        exon = row['exon']
        wt = row['wtseq']
        ed = row['edseq']

        workdir = '%s/temp/%s/%s/%s' % (os.getcwd(), record.genesym, pe_system, exon)
        os.makedirs(workdir, exist_ok=True)

        dp = prd.DeepPrime(ID, wt, ed, 'sub', 1, silence=True, gpu=0)
        if dp.pegRNAcnt == 0: continue

        totalpegRNAs += dp.pegRNAcnt

        df_dpscores = dp.predict('PE2max')
        sorted_df = df_dpscores.sort_values(by='PE2max_score', ascending=False)

        sorted_df.to_csv('%s/%s.csv' % (workdir, index))

        # if ID not in dict_top10:
        # dict_top10[ID] = scores
    # loop END:

    print('totalpegRNAs', totalpegRNAs)
# def END: run_dp


def main_run(analysistag):
    ## Set Parameters ##
    data_dir    = '%s/data/%s' % (sBASE_DIR, analysistag)
    ref         = '%s/ref' % sBASE_DIR
    genome      = 'GRCh38'
    fa          = '%s/%s.fa' % (ref, genome)
    fasta       = genet.utils.Fasta(fa)

    options     = dict([readline.strip('\n').split(': ') for readline in open('%s/all_inputs.txt' % data_dir)])

    pe_system   = options.get('PE_Type', '')
    if not pe_system: sys.exit('Invalid PE system: Please check your input parameters')


    edittype    = options.get('EditType', '')[:3]
    editsize    = options.get('EditLength')

    pbs_min     = int(options.get('PBSmin', '7'))
    pbs_max     = int(options.get('PNSmax', '15'))
    rtt_max     = int(options.get('RTTmax', '40'))
    if not 0 <= rtt_max <= 40: sys.exit('Invalid RTT Size %s: Please check your input parameters' % rtt_max)

    top_view       = int(options.get('TopView', '200'))
    marker         = options.get('MarkerPriority', '')
    intron         = options.get('IntronOverhang', 'T')

    frame          = '' ## add to webtool

    inputtype      = options.get('InputType', '')
    input1         = options.get('inputbox1', '')
    input2         = options.get('inputbox2', '')

    if not input1: sys.exit('Invalid Input Empty Field: Please check your input parameters')
    if input2 == '' : sys.exit('Invalid Input Empty Field: Please check your input parameters')

    flank          = 60

    if inputtype in ['GeneSym', 'NMID', 'EnsemblID', 'HGNC']:

        genesym, chrID, strand, coord = get_entrez_inquiry_from_file(ref, fasta, inputtype, input1, input2)
        base         = genesym
        target       = int(input2)
        df           = make_dp_input_v2(fasta, pe_system, chrID, strand, coord, flank, target)
        dict_refAA   = genet.utils.RefAA(genesym).dict_refAA

    elif inputtype in ['ClinVar']:
        base         = input1
        target       = input2 # model or therapy for ClinVar and COSMIC
        df, editinfo, dict_refAA = df_make_dp_input_clinvar (ref, fasta, input1, input2, flank)
        if editinfo.startswith('Invalid'):
            sys.exit(editinfo)

    elif inputtype in ['COSMIC']:
        base         = input1
        target       = input2 # model or therapy for ClinVar and COSMIC
        df, editinfo, dict_refAA = df_make_dp_input_cosmic (ref, fasta, input1, input2, flank)


    elif inputtype in ['Sequence']:
        base       = 'seq.%s' % analysistag[:10]
        target     = inputtype
        df         = df_make_dp_input_seq (input1, input2, target,  flank)
        dict_refAA = {}


    elif inputtype in ['Position']:
        base    = 'pos.%s' % analysistag[:10]
        target  = inputtype
        df      = df_make_dp_input_pos (fasta, input1, input2, target, flank)
        dict_refAA = {}


    tempdir     = '%s/temp/%s/%s' % (sBASE_DIR, base, pe_system)
    os.makedirs(tempdir, exist_ok=True)

    dict_opts = {'pbs_max':    pbs_max,
                 'pbs_min':    pbs_min,
                 'rtt_max':    rtt_max,
                 'pe':         pe_system.split('-', 1)[0],
                 'celltype':   pe_system.split('-', 1)[1],
                 'edittype':   edittype,
                 'editsize':   editsize,
                 'base':       base,
                 'target':     target,
                 'top':        top_view,
                 'marker':     marker,
                 'intron':     intron,
                 'frame':      frame,
                 'inputtype':  inputtype,
                 }

    model_info  = models.LoadModel('DeepPrime', dict_opts['pe'], dict_opts['celltype'])
    model_dir   = model_info.model_dir
    results     = '%s/results' % data_dir
    os.makedirs(results, exist_ok=True)

    mp_run_dp(tempdir, model_dir, results, df, dict_opts)
    mp_snv_marker(analysistag, tempdir, model_dir, results, dict_opts, dict_refAA)
    os.system('ln -s %s/%s_html.csv %s/Results_forHTML.csv' % (results, base, data_dir))
# def END: main


def get_entrez_inquiry_from_file (ref, fasta, inputtype, input, target):

    idfile  = '%s/forwebtool/%s/%s' % (ref, inputtype, input)
    inf     = open(idfile, 'r')
    gi, genesym, chrom, strand, exons = inf.readline().strip('\n').split('\t')
    inf.close()

    exoncnt    = len(exons.split(','))

    if int(target) == 0:
        coord = dict([pos.split('|') for pos in exons.split(',')])

    else:
        try: coord   = dict([pos.split('|') for pos in exons.split(',')])[str(target)]
        except KeyError:
            sys.exit('Exon Number is Invalid: Number of exons in CDS of %s= %s' % (input, exoncnt))

    return genesym, chrom, strand, coord
#def END: get_entrez_inquiry_from_file


def preprocess_COSMIC():

    # Index COSMIC Database ##
    ref      = '%s/ref' % sBASE_DIR
    indexdir = '%s/cosmic_temp' % ref
    os.makedirs(indexdir, exist_ok=True)

    infile   = '%s/cosmic_mutations_hg38.tsv.gz' % ref
    dict_cos = parse_cosmic(infile)
    list_id  = sorted(list(dict_cos.keys()))
    print(len(list_id))
    #
    # dict_notation = {}
    # altnotations  = df['altnotation']
    # for altnote in altnotations:
    #     if altnote not in dict_notation:
    #         dict_notation[altnote] = 0
    #     dict_notation[altnote] += 1
    #
    # snvcnt = 0
    # notsnv = 0
    # for note, cnt in dict_notation.items():
    #     if len(note) == 3:
    #         snvcnt += cnt
    #         print(note, cnt)
    #     else:
    #         notsnv += cnt
    # print(snvcnt)
    # print(notsnv)

    bins      = 300
    total     = len(list_id)
    list_bins = [[int(total * (i + 0) / bins), int(total * (i + 1) / bins)] for i in range(bins)]

    outfile_index = '%s/index.txt' % indexdir
    outf          = open(outfile_index, 'w')

    for start, end in list_bins:
        list_keys = list_id[start:end]

        outtag         = '%s-%s'          % (list_keys[0], list_keys[-1])

        print(outtag)
        outfile_pickle = '%s/%s_data.tmp' % (indexdir, outtag)

        out       = '%s\t%s\n' % (outtag, outfile_pickle)
        outf.write(out)

        dict_out  = {key:dict_cos[key] for key in list_keys}
        outfile   = open(outfile_pickle, 'wb')
        pickle.dump(dict_out, outfile)
        outfile.close()
    #loop END: start, end
    outf.close()
#def END: preprocess_clinvar



def preprocess_clinvar (tmpdir, infile):

    dict_vcf = parse_vcf_file(infile)
    list_id  = sorted([id for id in list(dict_vcf.keys())])
    print(len(list_id))

    bins      = 150
    total     = len(list_id)
    list_bins = [[int(total * (i + 0) / bins), int(total * (i + 1) / bins)] for i in range(bins)]


    outfile_index = '%s/index.txt' % tmpdir
    outf          = open(outfile_index, 'w')

    for start, end in list_bins:
        list_keys = list_id[start:end]

        outtag         = '%s-%s'          % (list_keys[0], list_keys[-1])
        print(outtag)
        outfile_pickle = '%s/%s_data.tmp' % (tmpdir, outtag)
        out             = '%s\t%s\n' % (outtag, outfile_pickle)
        outf.write(out)

        dict_out  = {key:dict_vcf[key] for key in list_keys}
        outfile   = open(outfile_pickle, 'wb')
        pickle.dump(dict_out, outfile)
        outfile.close()
    #loop END: start, end
    outf.close()

    pass
# def END: preprocess_COSMIC




def main():

    #preprocess_COSMIC()
    #preprocess_clinvar()
    pass
# def END: main



if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        function_name = sys.argv[1]
        function_parameters = sys.argv[2:]
        if function_name in locals().keys():
            locals()[function_name](*function_parameters)
        else:
            sys.exit('ERROR: function_name=%s, parameters=%s' % (function_name, function_parameters))
    # if END: len(sys.argv)
# if END: __name__
