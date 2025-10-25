# *Processus Universalis* annotated XML corpus
This repository contains XML documents from the 2018-2019 "Processus Universalis" project by Netzwerk Alchemie (Gotha) and serves as a testbed to explore future applications building on this chemically annotated recipe corpus data.

## The project
Certain procedures in historical recipe groups, particularly those of significant interest, became the focus of extensive reworking and circulation. One such cluster of texts is the *Processus Universalis*, which exists in over 100 known manuscript versions (Gelius 1996, Moenius et al. 2023).
Since 2018, the [Gotha Alchemy Network](https://www.uni-erfurt.de/en/gotha-research-centre/research/working-groups-and-networks/alchemy-network) has undertaken a comprehensive study of this group. Historians of chemistry involved in the project have transcribed 17 versions of the *Processus Universalis* and catalogued many more (Moenius et al. 2023). These texts have been systematically analysed from historical, archival, and chemical perspectives. A substantial number of recipes were also reproduced in laboratory conditions to assess their chemical plausibility, functionality, and points of failure. This collaborative endeavour culminated in a large edited volume that brings together the findings of this scholarly and experimental effort.

By 2019, the project had begun to explore computational approaches to textual analysis, including the use of stylometry to group similar recipe variants (within the *Processus Universalis* group, forthcoming edited collection). These early experiments succeeded in replicating groupings previously established through chemical and historical scholarship (Werthmann/Wunderlich 2023). While such findings may appear to merely confirm existing knowledge, it remains striking how a computational tool that only had access to language data was able to reproduce conclusions that historians had reached relying mainly on their chemical knowledge. 


<img alt="Network output of R Stylo analysis" src="https://github.com/sarahalang/processus-universalis/blob/main/_edersDelta_100-1000mfw_stylo-network_.png" width="50%">
<img alt="R Stylo CA, Eder's Delta" src="https://github.com/sarahalang/processus-universalis/blob/main/_processusUniversalis2025_CA_300_MFWs_Culled_0__Eder's%20Delta_.png" width="50%">
<img alt="R Stylo MDS" src="https://github.com/sarahalang/processus-universalis/blob/main/_processusUniversalis2025_MDS_300_MFWs_Culled_0_.png" width="50%">

However, these digital experiments (such as using R Stylo, Eder/Rybicki/Kestemont 2016) were initially constrained by the linguistic nature of the source material. The texts are primarily written in Early Modern German, often interspersed with Latin, and lacked standardised orthography. Consequently, most natural language processing tools available at the time performed poorly. Text reuse analysis, using tools such as [Jonathan Reeve’s text matcher](github.com/JonathanReeve/text-matcher), did yield some results and `text matcher`remains a valuable resource in the digital humanities due to its compatibility with historical language variants. Nonetheless, the visualisation and interpretation of text reuse proved challenging: The extent of textual overlap among the *Processus Universalis* manuscripts is such that traditional linear comparison methods are insufficient; network-based representations are more appropriate, though they complicate textual comparison. Dating the texts also presents a difficulty: while a few manuscripts are datable, the majority are not, limiting chronological analysis. Despite these constraints, the project has yielded a valuable corpus for future research. 

In addition to the 17 fully transcribed texts, an XML dataset exists in which scholars have manually annotated and categorised key structural components of the recipes. These annotations identify various steps in the chemical processes, noting which authors reference which components and which omit them. This structured data in key–value format is  available, but it was not yet fully utilised in earlier stages of the project. 
The combination of chemically validated data (Werthmann/Wunderlich 2023), manual expert-verified textual annotation, and more advanced digital tools suggests a new phase of scholarship may be possible, one that integrates computational analysis to uncover patterns and insights not accessible through traditional methods alone.


## The data
As explained above, the *Processus Universalis* recipe group appears in over fifty texts, seventeen of which have been fully transcribed and annotated by human experts with information about the chemical processes and procedural stages described and extensively studied (experimentally, digitally and traditionally), resulting in the availability of ample contextual information. In particular, this can be found in the detailed studies (articles) in which scholars of the Gotha Alchemy Network document how they interpreted the historical texts, derived experimental hypotheses, and validated these through laboratory replication. This provides a uniquely detailed ground truth for tracing the complete interpretative process from text to experiment and verified chemical result. Since then, computational methods have advanced considerably, allowing us to build directly on this well-prepared dataset, a goldmine yet untapped.

The XML data include both full transcriptions and key-value annotations standardised across the corpus, providing a human-generated ground truth (graciously provided by Alexander Kraft and Thomas Moenius) to train and evaluate any algorithms.

Such validated data are essential for replicating this interpretative process computationally to extract the underlying chemistry from alchemical language. Given the complexity of this task, having human-verified results for this recipe group will be of exceptional value, serving as a sandbox and testbed for developing and evaluating AI methods to extract the underlying chemistry from alchemical language.

So far, the data in `sammlung_aller_texte.xml` are in a custom XML format (with annotations in German) and intended for internal use. This dataset can later be modified for publication in a more standardised format for externals to use. Here is a code example showing the beginning of text G1-A1, containing the transcribed text and some examples of the chemical annotations standardised across the corpus:
```
<div type="g1a1" n="A1 Höchster Schatz und Kleinod der Welt" when="">
        <head>A1 Höchster Schatz und Kleinod der Welt</head> 1 <head>A. </head>
        <keys type="Zuschreibung der Vorschrift" n=" FEHLT;"/>
        <head>Höchster Schatz und Kleinod der Welt Den gerechten und wahrhaften Lapidem philosophorum
            oder Tincturam universalem, medicinalem et metallurgicam ex regno universalissimo zu
            machen. </head>
        <keys type="Philosophische Vorbetrachtung" n=" FEHLT;"/>
        <head>Vorarbeit oder Praeparatio Salis terrae, seu nitri nostri philosophorum, e terra
            Virginea. </head>
        <keys type="Datierung der Probenahme"
            n=" Mai; klarer Himmel; kein Wind; kein Regen; früh morgens; bei Sonnenaufgang;"/>
        <keys type="Ort der Probenahme" n=" schöne Wiese; Wiese mit Blumen; "/>
        <keys type="Art der Erde"
            n=" fette schwarze Erde; rote Erde; gelbe Erde; lehmig; bolarisch;"/>
        <keys type="Art der Probenahme"
            n=" knietief; ohne Gras; ohne Wurzeln; ohne Steine; 20-24 Zentner;"/>
        <keys type="Aufteilung der Erde" n=" FEHLT;"/>
        <keys type="Fundort der Erde" n=" FEHLT;"/> Im Monath May, wenn der Himmel gantz klar, hell
        und rein, auch das Gewitter gantz still, ohne einigen Wind und Regen, auch die Lufft voll
        lieblichen Geruchs ist, daß gleichsam die Luft, wenn mann darein sieht, von feinen lieblichen
        und schwüligen Dämpfen brodent oder rauchet, soll mann morgensfrüh bey Sonnen-Aufgang, auf
        eine schöne Wiese gehen, die eine gute, fette schwartze, viel beßer aber roth oder gelbe
        Bullarische das ist balligte Erden hat, oder die leimigt ist, und sonsten gleichfalls von
        Natur von allerhand wohlriechenden Blumen zu tragen pfleget, darauf soll mann etliche große
        weit und runde Graben machen, und in die Weite und Länge ohngefähr 5 oder 4 Maßruthen breit
        graben laßen, bis an die Knie tief, das Gras und die Wurtzeln aber müßen vorhero alle daraus
        abgesondert werden und kann mann solches mit den Wasen erstlich stück weis mit einem
        Grabscheid ausstechen, hernach die Grube mit anderer Erde und etwas Mist wieder zu füllen,
        oder als denn den Wasen wieder darauf setzen, so giebt die ausgegrabene Erde der Wiese ein
        geringsten kleinen Schaden <keys type="Imprägnation der Erde"
            n=" Erde ausbreiten; 14 Tage; Einwirkung von Gestirn und Himmel; bei Regen abdecken;"/>
        <keys type="" n=""/>
```
Be aware that the text and group numbering was changed throughout the project, which is why the .txt files used in R stylo have the newer naming convention (E groups) whereas the XML file still follows the old naming convention (A groups). 

## References
Eder, Maciej, Jan Rybicki und Mike Kestemont (2016): „Stylometry with R: A Package for Computational Text Analysis“. In: The R Journal. 8 (1), 107–121.
GitHub: https://github.com/computationalstylistics/stylo

Gelius, Rolf. "Der «Processus universalis» nach Michael Sendivogius: Zur Entstehungsgeschichte einer neuzeitlichen Variante des alchimischen Grossen Werkes." Gesnerus 53, no. 3-4 (1996): 183-193.

Thomas Moenius, Alexander Kraft, Gerhard Görmar: Andreas Orthelius und der Processus Universalis, in: S. Lang (ed.) Alchemische Labore: Texte, Praktiken und materielle Hinterlassenschaften (Graz 2023), pp. 329-349. DOI: https://doi.org/10.25364/978390337404117 

Rainer Werthmann, Christian-Heinrich Wunderlich: Eine Rekonstruktion alchemischer Laborprozesse: am Beispiel der Processus Universalis Rezeptgruppe, in: S. Lang (ed.) Alchemische Labore: Texte, Praktiken und materielle Hinterlassenschaften (Graz 2023), pp. 351-362. DOI: https://doi.org/10.25364/978390337404118 

Forthcoming edited collection, Halle Museum für Vorgeschichte and Netzwerk Alchemie (led by Martin Mulsow), likely to appear 2026. 

