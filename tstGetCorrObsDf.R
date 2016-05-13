setwd("~/Documents/Work/DataScience/Kaggle/StateFarm")

require(ggplot2)

tstGetCorrObsDf = read.csv('data/img_01_import_data_SFDD_ImgSz_64_tstGetCorrObsDf.csv')

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL, fileName = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    df <- df[, c(measure, dim)]

    pltDf <- tidyr::gather_(df, 'key', 'value', gather_cols = measure)
    if (!is.null(highLightIx))
        bstDf <- tidyr::gather_(df[highLightIx, ], 'key', 'value',
                                gather_cols = measure)

    if (nrow(pltDf) > 0) {
        gp <- ggplot(pltDf, aes_string(x = dim[1], y = 'value'))

        if (length(dim) > 1) {
            aesStr <- sprintf("color = as.factor(%s)", dim[2])
            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
            gp <- gp + geom_line(mapping = aesMap)

            if (length(dim) <= 2)
                aesStr <- sprintf("color = as.factor(%s)", dim[2]) else
                    aesStr <- sprintf("color = as.factor(%s), shape = as.factor(%s)", dim[2], dim[3])
            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
            gp <- gp + geom_point(mapping = aesMap)
        } else
            gp <- gp + geom_line(color = 'blue') + geom_point(color = 'blue')

        if (!is.null(scaleXFn) &&
            !is.null(scaleXFn[dim[1]])) {
            gp <- gp + switch(scaleXFn[dim[1]],
                              log10 = scale_x_log10(),
                              stop("switch error in mypltModelStats"))

            if (scaleXFn[dim[1]] == "log10") {
                #print("scaleXFn is log10")
                if (0 %in% unique(df[, dim[1]]))
                    for (key in measure) {
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                    }
            }
        }

        gp <- gp +
            ylab('') +
            scale_linetype_identity(guide = "legend") +
            theme(legend.position = "bottom")

        if (length(dim) < 3)
            gp <- gp + facet_grid(. ~ key,
                                  scales = "free", labeller = "label_both")
        gp <- gp + facet_grid(as.formula(paste(paste0(tail(dim, -2), collapse = "+"),
                                               "~ key")),
                              scales = "free", labeller = "label_both")

        if (!is.null(title))
            gp <- gp + ggtitle(title)

        if (!is.null(highLightIx))
            for (key in measure)
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)

    }

    if (!is.null(fileName)) {
        savGP <- gp
        png(filename = fileName, width = 480 * 1, height = 480 * 1)
        print(gp)
        dev.off()

        gp <- savGP
    }

    return(gp)
}

mypltModelStats(subset(tstGetCorrObsDf, !(xRowsN %in% c(5, 10))), c('mean', 'median', 'duration'),
                dim = c('chunkSize', 'yRowsN', 'xRowsN'),
                scaleXFn = NULL,
                highLightIx = which(tstGetCorrObsDf$bestFit == 'True'))