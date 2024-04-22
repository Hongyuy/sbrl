# This function sbrl() does the training work
sbrl <- function(tdata, iters=30000, pos_sign="1", neg_sign="0", rule_minlen=1, rule_maxlen=1, minsupport_pos=0.10, minsupport_neg=0.10, lambda=10.0, eta=1.0, alpha=c(1,1), nchain=10) {
    
    #require(arules)
    # separate data by the label signs
    pos_data <- tdata[tdata$label==pos_sign,]
    neg_data <- tdata[tdata$label==neg_sign,]
    # using eclat algorithm in arules package to extract rules for positive/negative data
    pos_rules <- arules::eclat(subset(pos_data, select =- label), parameter = list(minlen=rule_minlen, maxlen=rule_maxlen, supp=minsupport_pos))
    neg_rules <- arules::eclat(subset(neg_data, select =- label), parameter = list(minlen=rule_minlen, maxlen=rule_maxlen, supp=minsupport_neg))
    # extract the featurenames, rulenames and the binary feature-rule matrices
    pos_featurenames <- attributes(attributes(pos_rules)$items)$itemInfo$labels
    pos_rulenames <- as(pos_rules, "data.frame")$items #inspect(pos_rules)$items
    pos_mat <- attributes(attributes(pos_rules)$items)$data
    neg_featurenames <- attributes(attributes(neg_rules)$items)$itemInfo$labels
    neg_rulenames <- as(neg_rules, "data.frame")$items #inspect(neg_rules)$items
    neg_mat <- attributes(attributes(neg_rules)$items)$data
    
    # extend the feature-rule matrices by merging pos/neg features
    pos_mat2 <- rbind(pos_mat, matrix(0, length(setdiff(neg_featurenames, pos_featurenames)), ncol(pos_mat)))
    neg_mat2 <- rbind(neg_mat, matrix(0, length(setdiff(pos_featurenames, neg_featurenames)), ncol(neg_mat)))
    
    pos_combined_featurenames <- c(as.character(pos_featurenames),as.character(setdiff(neg_featurenames, pos_featurenames)))
    neg_combined_featurenames <- c(as.character(neg_featurenames),as.character(setdiff(pos_featurenames, neg_featurenames)))
    featurenames <- sort(pos_combined_featurenames, index.return=TRUE) # all the features
    # indices for the positive features
    pos_idx <- featurenames$ix
    featurenames <- featurenames$x
    # indices for the negative features
    neg_idx <- sort(neg_combined_featurenames, index.return=TRUE)$ix
    
    pos_mat3 <- pos_mat2[pos_idx, ]
    neg_mat3 <- neg_mat2[neg_idx, ]
    # now we had the rows correct. let's fix the columns (rules)
    rulenames <- c(as.character(pos_rulenames), as.character(neg_rulenames))
    idx <- order(rulenames)[!duplicated(sort(rulenames))]
    rulenames <- rulenames[idx]
    labelnames <- c("{label=0}", "{label=1}")
    # get the columns correct for feature_rule matrix
    mat <- as.matrix(cbind(pos_mat3, neg_mat3)[, idx])
    
    # now the columns are correct. for feature_rule matrix
    mat_data_feature <- get_data_feature_mat(tdata, featurenames)
    # get the data_rule matrix by multiplying data_feature and feature_rule matrices
    mat_data_rules <- mat_data_feature %*% mat
    mat_data_rules <- t(t(mat_data_rules)>=c(colSums(mat)))+0
    #t(t(head(mat_data_rules))>=c(colSums(mat)))+0
    
    # print each rule with a trailing binary string representing whether each data point is captured by that rule.
    # also print the labels in this format
    out_file=tempfile()
    label_file=tempfile()
    #cat(sprintf("[debug] created out_file=%s, size=%s\n", out_file, file.size(out_file)))
    #cat(sprintf("[debug] created label_file=%s, size=%s\n", label_file, file.size(label_file)))
    write.table(as.matrix(t(mat_data_rules)), file=out_file, sep=' ', row.names=rulenames, col.names=FALSE, quote=FALSE)
    label <- t(cbind((tdata$label==neg_sign) +0, (tdata$label==pos_sign) +0))
    write.table(as.matrix(label), file=label_file, sep=' ', row.names=labelnames, col.names=FALSE, quote=FALSE)
    
    # call the C functions through Rcpp wrapper function sbrl_train
    rs<-.Call('sbrl_train', PACKAGE = 'sbrl', 0, 0, list(lambda, eta, 0.5, alpha, iters, nchain), out_file, label_file,
              rulenames, labelnames, as.matrix(t(mat_data_rules)), as.matrix(label))$rs
    #cat(sprintf("[debug] written out_file=%s, size=%s\n", out_file, file.size(out_file)))
    #cat(sprintf("[debug] written label_file=%s, size=%s\n", label_file, file.size(label_file)))
    unlink(out_file)
    unlink(label_file)
    
    structure(list(rs=rs, rulenames=rulenames, featurenames=featurenames, mat_feature_rule=mat), class="sbrl")
}

# This funtion predicts the class-0, class-1 probabilities given the sbrl-model and data
predict.sbrl <- function(object, tdata, ...) {
    # comment these lines
    mat_data_feature <- get_data_feature_mat(tdata, object$featurenames)
    mat_data_rules <- mat_data_feature %*% object$mat_feature_rule
    mat_data_rules <- t(t(mat_data_rules)>=c(colSums(object$mat_feature_rule)))+0
    nrules <- ncol(object$mat_feature_rule)
    nsamples <- nrow(tdata)
    mat_idx <- matrix(0, nrow = nsamples, ncol = nrules)
    for (i in 1:(nrow(object$rs)-1)) {
        mat_idx[, object$rs$V1[i]] = i
    }
    mat_satisfy <- mat_data_rules * mat_idx
    
    # find the earliest rule that captures the data
    mat_caps <- as.matrix(apply(mat_satisfy, 1, function(x) ifelse(!identical(x[x>0], numeric(0)), min(x[x>0]), NaN) ))
    mat_caps[is.na(mat_caps)] = nrow(object$rs)
    mat_prob <- as.double(object$rs$V2)[mat_caps]
    list(V1=1-mat_prob, V2=mat_prob)
}

# S3 methods.
# print the model in an interpretable way (if ... then ...)
print.sbrl <- show.sbrl <- function(x, useS4 = FALSE, ...) {
    cat(sprintf("The rules list is : \n"))
    for (i in 1:nrow(x$rs)) {
        if (i==1)
        cat(sprintf("If      %s (rule[%d]) then positive probability = %.8f\n", x$rulenames[x$rs$V1[i]], x$rs$V1[i], x$rs$V2[i]))
        else if (i==nrow(x$rs))
        cat(sprintf("else  (default rule)  then positive probability = %.8f\n", x$rs$V2[nrow(x$rs)]))
        else
        cat(sprintf("else if %s (rule[%d]) then positive probability = %.8f\n", x$rulenames[x$rs$V1[i]], x$rs$V1[i], x$rs$V2[i]))
    }
}

# This function gets the data-by-feature matrix, given the data and all the feature names
get_data_feature_mat <- function(data, featurenames) {
    mat_data_feature <- matrix(0, nrow=nrow(data), ncol=length(featurenames))
    for (i in 1:length(featurenames)) {
        feature <- featurenames[i]
        conds <- strsplit(feature, '=')
        mat_data_feature[which(data[,conds[[1]][1]]==conds[[1]][2]), i] <- 1
    }
    mat_data_feature
}
