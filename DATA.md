# Data

The [complete data package](https://drive.google.com/drive/folders/1qMdv_NxpsLZlu510izs26V6b02smGAoH?usp=drive_link)
includes large benchmark files not stored in this repository. Preserve its
directory structure when placing benchmark data under `data/`. The Git repository tracks a partial copy of the
benchmark data (DS1-DS4 UFBoot and MrBayes files).

<!-- AUTHOR TODO: verify that the Drive folder is publicly accessible and that
its contents match the layout described below. Consider depositing the data
package in a permanent archive (e.g., Zenodo or Dataverse) and adding its DOI
here before acceptance. -->

## Data dictionary

| Path | Format | Contents | Citation |
|---|---|---|---|
| `data/hohna_datasets_fasta/DS1.fasta` - `DS8.fasta` | FASTA (plain text, wrapped at 60 characters per line) | Eight empirical nucleotide alignments | DS1-DS8 benchmark datasets of Lakner et al. (2008, *Syst. Biol.*, DOI: 10.1080/10635150801886156); see also Höhna & Drummond (2012, DOI: 10.1093/sysbio/syr074) and Whidden & Matsen (2015, DOI: 10.1093/sysbio/syv006) |
| `data/ufboot_data_DS1-8/<dataset>/<dataset>_ufboot_rep_<1..10>` | Newick (plain text), one rooted tree per line | UFBoot topology-support trees: 10 replicates of 10,000 maximum-likelihood bootstrap trees each | IQ-TREE UFBoot (Minh et al. 2013, *Mol. Biol. Evol.*), following Zhang & Matsen (2019); replicated here as in VBPI (Zhang 2023) |
| `data/raw_data_DS1-8/<dataset>/rep_<1..10>/<dataset>.trprobs` | NEXUS tree file (plain text); each tree line carries an `[ID: ...]` comment | MrBayes posterior samples of tree topologies with posterior probability weights; 10 replicate files per dataset | MrBayes 3.2 runs (Ronquist et al. 2012, *Syst. Biol.*); weights are combined with `samp_size = 750001` |
| `data/raw_data_DS1-8/<dataset>/emp_tree_freq.pkl` | Python pickle (binary) | Precomputed dictionary mapping a tree-split key (tuple of bit-array integers) to its empirical topology frequency. The workflow recomputes these frequencies from the `.trprobs` files at run time and does not read this pickle. The DS2 copy is an empty placeholder (`None`); DS1, DS3, and DS4 contain 2611, 327, and 10677 entries. | Derived from the `.trprobs` files |
| `<MRBAYES_TREE_ROOT>/tree<1..42>/tree<1..42>.nex.t` | NEXUS tree log (plain text) | Long-chain fixed-topology MrBayes samples: ground-truth branch-length distributions for the 42 DS1 posterior topologies. Set `MRBAYES_TREE_ROOT` to the parent directory. | Long MrBayes 3.2 run: 4 chains, one billion iterations, sampled every 100 iterations |
| `synthetic_exp/L8trees/seqlen_<length>/tree_<replicate>/dna.fasta` | FASTA (plain text) | Simulated alignment for eight taxa `L1`-`L8` | Author-generated: Jukes-Cantor simulation on a random eight-taxon tree with Exp(10) branch lengths |
| `synthetic_exp/L8trees/seqlen_<length>/tree_<replicate>/tree.newick` | Newick (plain text) | The generating (reference) tree for the paired alignment | Author-generated, as above |
| `synthetic_exp/ufboot/L8ALL_ufboot_rep_<1..10>` | Newick (plain text), one tree per line | Shared topology support: 10 replicates of 10,395 trees each | IQ-TREE UFBoot (Minh et al. 2013, *Mol. Biol. Evol.*), pooled over all synthetic datasets (`L8ALL`) |

## Benchmark Data

### Alignments

`data/hohna_datasets_fasta/` contains the DS1-DS8 alignments. The number of
taxa and alignment sites are:

| Data set | DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 |
|---|---|---|---|---|---|---|---|---|
| Taxa | 27 | 29 | 36 | 41 | 50 | 50 | 59 | 64 |
| Sites | 1949 | 2520 | 1812 | 1137 | 378 | 1133 | 1824 | 1008 |

<!-- AUTHOR TODO: confirm the FASTA copies correspond to the distribution used
for the manuscript analyses (the directory name refers to the Höhna
distribution; the datasets originate from Lakner et al. 2008). -->

### UFBoot support and MrBayes posterior files

`data/ufboot_data_DS1-8/` and `data/raw_data_DS1-8/` hold the UFBoot support
trees and the MrBayes `.trprobs` posterior tree files. Full UFBoot support and
raw MrBayes posterior files for all of DS1-DS8 are in the complete data
package; files in this checkout are a partial copy covering DS1-DS4. Figure 9
does not use these benchmark files.

For DS1 fixed-tree experiments (Figures 4-6 and Table 2), the workflows use
`data/hohna_datasets_fasta/DS1.fasta`, DS1 MrBayes `.trprobs` files, and UFBoot
support files. Figure 5/6 additionally requires fixed-tree MrBayes samples
`tree1/tree1.nex.t` through `tree42/tree42.nex.t`; set `MRBAYES_TREE_ROOT` to
their parent directory. These samples are separate from the `.trprobs` files
used to select and fit tree topologies.

## Figure 9: Synthetic Data

The posterior-recovery experiment uses 10 simulated eight-taxon trees. Branch
lengths follow `Exp(10)`; alignments are simulated under Jukes-Cantor at lengths
10, 20, 50, 100, 200, 500, and 1000 (70 alignments total).

- `L8trees/seqlen_<length>/tree_<replicate>/` contains `dna.fasta` (taxa
  `L1`-`L8`) and the generating `tree.newick`.
- `ufboot/L8ALL_ufboot_rep_<replicate>` contains 10 support files with 10,395
  Newick trees each; the workflow pools them as the topology support set.

## Terminology

- **Topology support**: the set of subsplit/rootsplit configurations with
  non-zero conditional probabilities, derived from the UFBoot trees and used
  to parameterize the SBN (`src/utils.py::get_support_from_mcmc`).
- **Tree-split key**: a tuple of bit-array integers encoding a tree topology,
  used as the dictionary key in `emp_tree_freq.pkl`.
- **Empirical topology frequency**: the MrBayes-estimated posterior
  probability of a topology, aggregated over the 10 `.trprobs` replicates with
  `samp_size = 750001` (`src/utils.py::summary`).
- **Fixed-tree branch samples**: one array per posterior topology holding
  positive branch-length samples from a fitted variational model (or the
  MrBayes reference), saved as `.npy` files under `results/fixed_tree/`.
