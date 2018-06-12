[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/Galaxy/introduction_to_galaxy_unilu.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/Galaxy/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/Galaxy/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Galaxy Introduction Exercise: From Peaks to Genes

**/!\ IMPORTANT NOTE:** This is an old version of the ["From peaks to genes"](http://galaxyproject.github.io/training-material/topics/introduction/tutorials/galaxy-intro-peaks2genes/tutorial.html) tutorial now provided by the [Galaxy Training Network](https://galaxyproject.org/teach/gtn/). For a much nicer version and many other Galaxy tutorials, check out the [Galaxy Training](http://galaxyproject.github.io/training-material/) webpage.

------

[![](https://github.com/ULHPC/tutorials/raw/devel/bio/galaxy/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/Galaxy/introduction_to_galaxy_unilu.pdf)

For a version of this tutorial with the results of important steps embedded and direct links to workflows, go to the Galaxy server, select "Shared Data" and then "Pages" from the top menu and have a look at [Galaxy Introduction](https://galaxy-server.uni.lu/u/sdiehl/p/galaxy-introduction).

## Scenario

We stumbled upon a paper ([Li et al., Cell Stem Cell 2012](http://www.sciencedirect.com/science/article/pii/S1934590912003347)) that contains the analysis of possible target genes of an interesting protein. The targets were obtained by ChIP-seq and the raw data is available through [GEO](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37268). The list of genes however is neither in the supplement of the paper nor part of the GEO submission. The closest thing we can find is a list of the regions where the signal is significantly enriched (peaks). The goal of this exercise is to turn this list of genomic regions into a list of possible target genes.

(Disclaimer: We are not affiliated with the authors of the paper and we don't make a statement about the relevance or quality of the paper. It is just a fitting example and nothing else.)

## Step 1: Upload peaks

Download the list of peaks (the file "GSE37268_mof3.out.hpeak.txt.gz") from GEO ([click here to get to the GEO entry](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37268)) to your computer. Use the upload button to upload the file to Galaxy and select "mm9" as the genome. Galaxy will automatically unpack the file.

This file is not in any standard format and just by looking at it, we cannot find out what the numbers in the different columns mean. In the paper they mention that they used the peak caller [HPeak](http://www.sph.umich.edu/csg/qin/HPeak/Readme.html). By looking at the HPeak manual we can find out that the columns contain the following information:

1. chromosome name*
2. start coordinate
3. end coordinate
4. length
5. location within the peak that has the highest hypothetical DNA fragment coverage (summit)
6. not relevant
7. not relevant

(*Note that the first column only contains the chromosome number, and X and Y are replaced by 20 and 21 for easier numerical manipulations.)

## Step 2: Get genes from UCSC

We also need a list of genes in mouse, which we can obtain from UCSC. Galaxy has the UCSC table browser integrated as a tool, so we don't need to download the data to our computers.

*    Tool: Get Data -> UCSC main table browser
*    Select clade "Mammal", genome "Mouse", assembly "mm9"
*    Select group "Genes and Gene Prediction Tracks", track "RefSeq Genes"
*    Select table "refGene"
*    Select region "genome"
*    Select output format "BED"
*    Click button "get output"
*    Click button "Send query to Galaxy"

## Step 3: Adjust chromosome naming

Have a look at both input files (either in the little preview window in the history or click on the eye icon to see one in the main frame) and find out what are the differences in the chromosome naming.

Apply the following workflow to GSE37268_mof3.out.hpeak.txt: Workflow 'Add "chr" at beginning of each line'.

After importing you can in the future use it by scrolling to the bottom of the tool panel, click on "All workflows" and then on the workflow name.

From carefully reading the HPeak manual, we should remember that it puts "20" and "21" instead of "X" and "Y". So now the chromosome names all start properly with "chr", but we still have "chr20" and "chr21" instead of "chrX" and "chrY".

*    Tool: Text Manipulation -> Replace text in a specific column
*    Input: result of workflow (Text reformatting on data X)
*    In colum: Column 1
*    Find pattern: chr20
*    Replace with: chrX
*    Do the same for "chr21" and "chrY", make sure you use the result of the first replacement as input (use rerun button and change input and search/replace)

Make sure the format of the output file is "interval", otherwise change it by clicking the pencil icon (do not convert to new format, but change data type).

## Step 4: Visualize peaks

To visualize the peaks it's best to convert them to BED format first, because most viewers cannot deal with interval (because interval format just exists in Galaxy).

*    Click on the pencil icon of the latest dataset
*    Under the header "Convert to new format" select "Convert Genomic Intervals to BED"
*    Click "Convert"
*    Look at the new dataset. Some columns with generic names have been added and others were removed to comply to BED format rules.
*    This generated a new dataset in BED format which we'll use for visualization. We will however continue to work with the interval dataset.

Display in IGB:

*    Go to the [IGB website](http://bioviz.org/igb/index.html)
*    Download and install the Integrated Genome Browser on your computer
*    Start IGB and in the right panel select species "Mus musculus" and  genome version "M_musculus_Jul_2007"
*    Go back to Galaxy
*    Click on the link "View" after "display with IGB" (expanded history view of BED dataset)
*    Type in your HPC credentials again, to allow IGB to access the data (you might also need to allow some connections and/or accept certificates)
*    Back in IGB, click "Load Data" next to the scroll bar on top to get to see the new track

## Step 5: Add promoter region to gene records

*    Tool: Operate on Genomic Intervals -> Get flanks
*    Input dataset: RefSeq genes from UCSC (UCSC Main on Mouse: refGene (genome))
*    Options: Region: "Around Start", Location: "Upstream",  Offset: 10000, Length: 12000

Inspect the resulting BED file and through comparing with the input find out what this operation actually did. Just look at the contents and compare the rows in the input to the rows in the output to find out how the start and end positions changed. Rename the dataset (by clicking on the pencil icon) to reflect your findings.

## Step 6: Find overlaps

*    Tool: Operate on Genomic Intervals -> Intersect
*    Return: Overlapping Intervals
*    of: result of step 5 (Get flanks on data X)
*    that intersect: result of step 3 (second Replace text)

The order of the inputs is important! We want to end up with a list of genes, so the corresponding dataset needs to be the first input.

## Step 7: Count genes on different chromosomes

To get a better overview of the genes we obtained, we want to look at their distribution across the different chromosomes.

*    Tool: Statistics -> Count occurrences of each record
*    Input: result from step 6 (Intersect on data X and data X)
*    Select column 1 (c1) with the chromosome names

## Step 8: Draw barchart

*    Tool: Bar chart (use tool search to find it)
*    Input: result of step 7
*    Use X Tick labels: Yes, column 2
*    Numerical column: c1
*    Plot title is up to you
*    Label for Y axis: number of genes

Galaxy has a second option to visualise tabular data, with built-in dynamic visualisations:

*    Expand the dataset view and click on the visualization icon
*    Choose "Charts"
*    Enter a chart title, e.g. "Genes on different chromsomes"
*    Select "Bar diagrams" -> "Regular"
*    On the top, click on "Add Data"
*    Enter a label, e.g. "count"
*    Values for x-axis: Column: 2 [str]
*    Values for y-axis: Column: 1 [int]
*    On the very top, click "Draw"

## Step 9: Name your history

In the history column click on "Unnamed history" at the top to rename it.

## Step 10: Make a workflow out of steps 6 to 8

*    Click on the history options and select "Extract workflow"
*    On the top click on "Uncheck all", then specifically check "Treat as input dataset" on GSE37268_mof3.out.hpeak.txt and UCSC Main on Mouse: refGene (genome), as well as "Include" on the Intersect, Count and Bar chart
*    Click "Create Workflow"

To make sure our workflow is correct we look at it in the editor and make some small adjustments.

*    Top menu: Workflow
*    Click on the name of your new workflow and select "Edit"

The individual steps are displayed as boxes and their outputs and inputs are connected through lines. When you click on a box you see the tool options on the right. Besides the tools you should see two additional boxes titled "Input dataset". These represent the data we want to feed into our workflow. Although we have our two inputs in the workflow they are missing their connection to the first tool (Intersect), because we didn't carry over the intermediate steps. Connect each input dataset to the Intersect tool by dragging the arrow pointing outwards on the right of its box (which denotes an output) to an arrow on the left of the Intersect box pointing inwards (which denotes an input). Connect each input dataset with a different input of Intersect.

You should also change the names of the input datasets to remember that the first one contains genes and the second one peaks. Don't forget to save it in the end by clicking on "Options" (top right) and selecting "Save".

## Step 11: Share workflow

Share your new workflow with the person to your left.

*    Top menu: Workflow
*    Click on your workflow's name and select "Share or publish"
*    Click "Share with a user"
*    Enter the username of the person to your left
*    Hit "Share"
*    Wait for the person on your right to do the same
*    Reload the workflows by clicking again on "Workflow" in the top menu
*    Under the header "Workflows shared with you by others" you should now see your right neighbour's workflow
*    Click on its name and select "View"
*    Compare with your workflow

## Step 12: Cleaning up

Download your workflow:

*    Top menu: Workflow
*    Click on your workflow's name and select "Download or Export"
*    Click on "Download workflow to file so that it can be saved or imported into another Galaxy server"
*    Save the workflow file on your computer

Clean up history:<br />
Delete all datasets that are neither initial input nor final results. Everything that can be easily recreated or is just part of an intermediate step can go. What I would keep are the extended genes, the intersect result and the bar chart (for a real analysis I would recommend to also download all final results). Deleted datasets can be undeleted for some time (see history options) and will only be ultimately removed from the server if they aren't used somewhere else or by somebody else and stay deleted for several weeks.

You can create new histories in the history options with "Create New".

To delete old histories:

*    History options: Saved histories
*    Check the history you want to delete
*    Click "Delete Permanently" on the bottom if you need to free up space or just "Delete"

## END
