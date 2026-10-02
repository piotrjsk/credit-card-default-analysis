# Mateusz Panek, Piotr Jasiak

# -------------------------------------------------------------------------
# ETAP 1: Wczytanie danych, EDA i Preprocessing
# -------------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(gridExtra)
library(ranger)

# Ustawienie globalnego ziarna na początku
set.seed(123)

# 1. Wczytanie zbioru danych
# Plik musi znajdować się w katalogu roboczym projektu
df <- read_csv("dane_do_projektu.csv") 

# Konwersja zmiennej celu na typ czynnikowy (factor) zgodnie z wymogami tidymodels
df <- df %>% 
  rename(default = last_col()) %>%  
  mutate(default = as.factor(default))

# -------------------------------------------------------------------------
# 2. Weryfikacja jakości danych i anomalii
# -------------------------------------------------------------------------
# Analiza rozkładu zmiennych kategorycznych w celu identyfikacji wartości 
# niezgodnych z dokumentacją (np. 0, 5, 6 w EDUCATION).

cat("--- Rozkład zmiennej EDUCATION ---\n")
print(table(df$EDUCATION))

cat("\n--- Rozkład zmiennej MARRIAGE ---\n")
print(table(df$MARRIAGE))

cat("\n--- Sprawdzenie zbalansowania klasy celu (default) ---\n")
print(table(df$default))
cat("Odsetek klasy pozytywnej (niespłacający): ", round(mean(as.numeric(as.character(df$default)) == 1) * 100, 2), "%\n")

# -------------------------------------------------------------------------
# 2.5 Eksploracja Danych (EDA) - Analiza skośności rozkładów
# -------------------------------------------------------------------------
# Wizualizacja uzasadniająca konieczność transformacji logarytmicznej 
# dla zmiennych kwotowych (LIMIT_BAL, BILL_AMT, PAY_AMT).

p1 <- ggplot(df, aes(x = LIMIT_BAL)) +
  geom_histogram(bins = 30, fill = "firebrick", color = "white") +
  labs(title = "Przed transformacją: Silna asymetria prawostronna", x = "Limit Kredytowy", y = "Liczność") +
  theme_minimal()

p2 <- ggplot(df, aes(x = log(LIMIT_BAL))) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Po transformacji: Rozkład zbliżony do normalnego", x = "Log(Limit)", y = "Liczność") +
  theme_minimal()

# Zestawienie wykresów
grid.arrange(p1, p2, ncol = 2)

# Wykres zależności default od statusu płatności we wrześniu (PAY_0)
pay0_plot <- df %>%
  mutate(default = ifelse(default == 1, "Tak (Default)", "Nie (Spłacone)")) %>%
  ggplot(aes(x = as.factor(PAY_0), fill = default)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("#2E8B57", "#D62728")) +
  labs(title = "Wpływ statusu płatności (PAY_0) na prawdopodobieństwo defaultu",
       x = "Status płatności w ostatnim miesiącu (PAY_0)",
       y = "Odsetek dłużników",
       fill = "Czy Default") +
  theme_minimal()

print(pay0_plot)

# -------------------------------------------------------------------------
# 3. Podział danych
# -------------------------------------------------------------------------
set.seed(123) # Kluczowe dla powtarzalności podziału
# Podział na zbiór treningowy i testowy (80/20) z zachowaniem stratyfikacji zmiennej celu
split <- initial_split(df, prop = 0.80, strata = default)
train_data <- training(split)
test_data  <- testing(split)

# -------------------------------------------------------------------------
# 4. Definicja Recipe
# -------------------------------------------------------------------------

base_rec <- recipe(default ~ ., data = train_data) %>%
  
  # A. Feature Engineering - Korekta anomalii
  # Scalanie niezidentyfikowanych kategorii w EDUCATION (0, 5, 6 -> 4)
  step_mutate(EDUCATION = ifelse(EDUCATION %in% c(0, 5, 6), 4, EDUCATION)) %>%
  # Korekta kategorii w MARRIAGE (0 -> 3)
  step_mutate(MARRIAGE = ifelse(MARRIAGE == 0, 3, MARRIAGE)) %>%
  
  # B. Konwersja zmiennych jakościowych na faktory
  step_mutate(SEX = as.factor(SEX),
              EDUCATION = as.factor(EDUCATION),
              MARRIAGE = as.factor(MARRIAGE)) %>%
  
  # C. Transformacja zmiennych numerycznych
  # Zastosowanie logarytmu ze znakiem (signed log) dla zmiennych o silnej asymetrii
  step_log(starts_with("BILL_AMT"), starts_with("PAY_AMT"), starts_with("LIMIT_BAL"), 
           offset = 1, signed = TRUE) %>%
  
  # D. Standaryzacja (wymagana m.in. dla SVM i metod regularyzacji)
  step_normalize(all_numeric_predictors()) %>%
  
  # E. Kodowanie zmiennych kategorycznych (Dummy Encoding)
  step_dummy(all_nominal_predictors())

# Weryfikacja poprawności receptury
prep_check <- prep(base_rec)
processed_data <- bake(prep_check, new_data = head(train_data))

cat("\nPodgląd struktury danych po przetworzeniu:\n")
glimpse(processed_data)


# -------------------------------------------------------------------------
# ETAP 2: Modelowanie i strojenie hiperparametrów
# -------------------------------------------------------------------------

set.seed(123) # Kluczowe dla powtarzalności walidacji krzyżowej
# Schemat walidacji krzyżowej (5-fold CV)
cv_folds <- vfold_cv(train_data, v = 5, strata = default)

# =========================================================================
# MODEL A: Elastic Net logistyczne (lasso i regresja grzbietowa)
# =========================================================================
cat("\n--- Trenowanie Elastic Net ---\n")

# Specyfikacja modelu
glmnet_spec <- logistic_reg(
  penalty = tune(), 
  mixture = tune()
) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

# Workflow
glmnet_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(glmnet_spec)

# Przestrzeń parametrów 
glmnet_grid <- grid_space_filling(
  penalty(), 
  mixture(), 
  size = 10
)

# Strojenie modelu
glmnet_res <- tune_grid(
  glmnet_wf,
  resamples = cv_folds,
  grid = glmnet_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

best_glmnet <- select_best(glmnet_res, metric = "roc_auc")
cat(paste("Najlepsze AUC dla Elastic Net:", round(show_best(glmnet_res, metric = "roc_auc", n=1)$mean, 4), "\n"))


# =========================================================================
# MODEL B: SVM (jądro Radial Basis Function)
# =========================================================================
cat("\n--- Trenowanie SVM (Radial) ---\n")

# Specyfikacja modelu
svm_spec <- svm_rbf(
  cost = tune(), 
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Workflow
svm_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(svm_spec)

# Siatka regularna (3 poziomy parametrów)
svm_grid_light <- grid_regular(
  cost(), 
  rbf_sigma(), 
  levels = 3 
)

set.seed(123) # Kluczowe dla powtarzalności osobnych foldów SVM
# Osobna walidacja CV dla SVM w celu optymalizacji czasu obliczeń (3-fold)
cv_folds_svm <- vfold_cv(train_data, v = 3, strata = default)

# Strojenie modelu
svm_res <- tune_grid(
  svm_wf,
  resamples = cv_folds_svm,
  grid = svm_grid_light,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

best_svm <- select_best(svm_res, metric = "roc_auc")
cat(paste("Najlepsze AUC dla SVM:", round(show_best(svm_res, metric = "roc_auc", n=1)$mean, 4), "\n"))


# =========================================================================
# MODEL C: Group Lasso
# =========================================================================
cat("\n--- Trenowanie Group Lasso ---\n")

# Przygotowanie macierzy danych (wymagane przez pakiet grpreg)
prepped_data <- prep(base_rec)
train_baked <- bake(prepped_data, new_data = NULL)

X_grp <- as.matrix(train_baked %>% dplyr::select(-default))
y_grp <- as.numeric(train_baked$default) - 1 # Konwersja factor -> 0/1

# Definicja grup zmiennych (grupowanie zmiennych powiązanych tematycznie)
varnames <- colnames(X_grp)
groups <- gsub("[_\\.].*", "", varnames) # Pobranie prefiksów
groups <- gsub("PAY[0-9]", "PAY", groups)       # Grupa statusów płatności
groups <- gsub("BILL.*", "BILL", groups)        # Grupa wyciągów
groups <- gsub("PAY.*AMT.*", "PAY_AMT", groups) # Grupa wpłat

library(grpreg)

set.seed(123) # Kluczowe dla powtarzalności walidacji w grpreg
# Walidacja krzyżowa dla Group Lasso
cv_grpreg <- cv.grpreg(
  X = X_grp, 
  y = y_grp, 
  group = groups, 
  penalty = "grLasso",
  family = "binomial",
  nfolds = 5
)

cat(paste("Min. błąd klasyfikacji (CV) Group Lasso:", round(min(cv_grpreg$cve), 4), "\n"))


# =========================================================================
# MODEL D: FDA (z MARS)
# =========================================================================
cat("\n--- Trenowanie FDA (MARS) ---\n")

library(discrim) 
library(earth)   

# Specyfikacja modelu z silnikiem 'earth' (MARS)
fda_spec <- discrim_flexible(
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth") %>%
  set_mode("classification")

fda_wf <- workflow() %>%
  add_recipe(base_rec) %>% 
  add_model(fda_spec)

# Definicja siatki parametrów
fda_grid <- grid_regular(
  num_terms(range = c(5, 30)), 
  prod_degree(),
  levels = 5
)

set.seed(123) # Kluczowe dla powtarzalności 
# Strojenie modelu
fda_res <- tune_grid(
  fda_wf,
  resamples = cv_folds,
  grid = fda_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

best_fda <- select_best(fda_res, metric = "roc_auc")
cat(paste("Najlepsze AUC dla FDA (MARS):", round(show_best(fda_res, metric = "roc_auc", n=1)$mean, 4), "\n"))


# =========================================================================
# MODEL E: Random Forest (Las Losowy)
# =========================================================================
cat("\n--- Trenowanie Random Forest ---\n")

# Specyfikacja modelu
rf_spec <- rand_forest(
  mtry = tune(),  # Liczba zmiennych losowanych w węźle
  min_n = tune(), # Minimalna liczba próbek w liściu
  trees = 300
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Workflow
rf_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(rf_spec)

# Siatka parametrów
rf_grid <- grid_regular(
  mtry(range = c(2, 10)),
  min_n(range = c(5, 20)),
  levels = 3
)

set.seed(123) # Kluczowe dla powtarzalności osobnych foldów RF
# Walidacja CV dla Random Forest (3-fold, analogicznie jak w SVM dla szybkości)
cv_folds_rf <- vfold_cv(train_data, v = 3, strata = default)

set.seed(123) # Kluczowe dla powtarzalności lasu losowego (bootstrap)
# Strojenie modelu
rf_res <- tune_grid(
  rf_wf,
  resamples = cv_folds_rf,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

best_rf <- select_best(rf_res, metric = "roc_auc")
cat(paste("Najlepsze AUC dla Random Forest:", round(show_best(rf_res, metric = "roc_auc", n=1)$mean, 4), "\n"))


# -------------------------------------------------------------------------
# ETAP 3: Ewaluacja końcowa na zbiorze testowym
# -------------------------------------------------------------------------

# A. Finalizacja modelu Elastic Net
final_glmnet_wf <- glmnet_wf %>% finalize_workflow(best_glmnet)
final_glmnet_fit <- final_glmnet_wf %>% fit(data = train_data)

results_glmnet <- predict(final_glmnet_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% dplyr::select(default)) %>%
  mutate(model = "Elastic Net")

# B. Finalizacja modelu SVM
final_svm_wf <- svm_wf %>% finalize_workflow(best_svm)
final_svm_fit <- final_svm_wf %>% fit(data = train_data)

results_svm <- predict(final_svm_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% dplyr::select(default)) %>%
  mutate(model = "SVM")

# C. Finalizacja modelu FDA (MARS)
final_fda_wf <- fda_wf %>% finalize_workflow(best_fda)
final_fda_fit <- final_fda_wf %>% fit(data = train_data)

results_fda <- predict(final_fda_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% dplyr::select(default)) %>%
  mutate(model = "FDA (MARS)")

# D. Finalizacja modelu Group Lasso
test_baked <- bake(prepped_data, new_data = test_data)
X_test_grp <- as.matrix(test_baked %>% dplyr::select(-default))
pred_grpreg <- predict(cv_grpreg, X = X_test_grp, lambda = cv_grpreg$lambda.min, type = "response")

results_grpreg <- tibble(
  .pred_0 = 1 - as.numeric(pred_grpreg),
  .pred_1 = as.numeric(pred_grpreg),
  default = test_data$default,
  model = "Group Lasso"
)

# E. Finalizacja modelu Random Forest
final_rf_wf <- rf_wf %>% finalize_workflow(best_rf)
final_rf_fit <- final_rf_wf %>% fit(data = train_data)

results_rf <- predict(final_rf_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% dplyr::select(default)) %>%
  mutate(model = "Random Forest")


# -------------------------------------------------------------------------
# PODSUMOWANIE WYNIKÓW
# -------------------------------------------------------------------------

all_results <- bind_rows(results_glmnet, results_svm, results_grpreg, results_fda, results_rf)

# Funkcja do metryk dla raportu
get_model_stats <- function(results_data, name) {
  auc_val <- results_data %>% roc_auc(truth = default, .pred_1, event_level = "second")
  
  f1_val <- results_data %>% 
    mutate(.pred_class = as.factor(ifelse(.pred_1 >= 0.3, 1, 0))) %>%
    f_meas(truth = default, estimate = .pred_class, event_level = "second")
  
  tibble(Model = name, AUC = auc_val$.estimate, `F1 Score` = f1_val$.estimate)
}

# Zestawienie
final_summary <- bind_rows(
  get_model_stats(results_glmnet, "Elastic Net"),
  get_model_stats(results_svm, "SVM"),
  get_model_stats(results_fda, "FDA (MARS)"),
  get_model_stats(results_grpreg, "Group Lasso"),
  get_model_stats(results_rf, "Random Forest")
) %>% arrange(desc(AUC))

cat("\n--- OSTATECZNY RANKING MODELI (AUC oraz F1 na zbiorze testowym) ---\n")
print(final_summary)

# Definicja kolorów dla modeli
model_colors <- c(
  "FDA (MARS)"    = "#2E8B57", # Ciemna zieleń (zwycięzca)
  "Elastic Net"   = "#1F77B4", # Niebieski
  "Group Lasso"   = "#FF7F0E", # Pomarańczowy
  "SVM"           = "#D62728", # Czerwony
  "Random Forest" = "#800080"  # Fioletowy
)

roc_plot <- all_results %>%
  group_by(model) %>%
  roc_curve(truth = default, .pred_1, event_level = "second") %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  # Linia odniesienia (losowy klasyfikator)
  geom_abline(lty = 2, color = "gray50", linewidth = 0.8) +
  # Krzywe ROC
  geom_path(linewidth = 1, alpha = 0.8) +
  # Skalowanie i kolory
  scale_color_manual(values = model_colors) +
  # Tytuły i osie
  labs(
    title = "Porównanie Wydajności Modeli (Krzywe ROC)",
    x = "1 - Specyficzność (Odsetek Fałszywych Alarmów)",
    y = "Czułość (Odsetek Wykrytych Dłużników)",
    color = "Model"
  ) +
  # Estetyka wykresu
  theme_light() +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  ) +
  # Wymuszenie proporcji 1:1
  coord_equal()

print(roc_plot)


# -------------------------------------------------------------------------
# ETAP 4: Analiza Ważności Zmiennych (Variable Importance)
# -------------------------------------------------------------------------

if(!require(vip)) install.packages("vip")
library(vip)

# Ekstrakcja silnika modelu Random Forest (ranger)
rf_obj <- extract_fit_engine(final_rf_fit)

# Wykres ważności predyktorów dla Random Forest
vip_plot <- vip(rf_obj, 
                num_features = 12, 
                geom = "col", 
                aesthetics = list(fill = "#800080", width = 0.7)) +
  # Dodanie etykiet tekstowych na słupkach
  geom_text(aes(label = round(Importance, 1)), 
            hjust = -0.2, size = 3.5, color = "#800080") +
  # Rozszerzenie osi
  expand_limits(y = c(0, max(vi(rf_obj)$Importance) * 1.1)) +
  labs(title = "Analiza ważności zmiennych VIP (Model Random Forest)", 
       y = "Wskaźnik istotności", 
       x = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold")
  )

print(vip_plot)

# Wyświetlenie statystyk modelu Random Forest
cat("\n--- Szczegóły silnika Random Forest (Ranger) ---\n")
print(rf_obj)

# -------------------------------------------------------------------------
# ETAP 5: Weryfikacja modelu: Interakcja PAY_0 i BILL_AMT1
# -------------------------------------------------------------------------

# Generowanie siatki prawdopodobieństwa (Partial Dependence Plot 2D)
pd_2d <- partial(rf_obj, 
                 pred.var = c("PAY_0", "BILL_AMT1"), 
                 grid.resolution = 30, 
                 train = train_baked,
                 prob = TRUE,
                 which.class = 2)

ggplot() +
  # WARSTWA 1: Mapa prawdopodobieństwa
  geom_tile(data = pd_2d, aes(x = PAY_0, y = BILL_AMT1, fill = yhat), 
            alpha = 0.7, width = 1) + 
  scale_fill_gradientn(colors = c("green3", "yellow", "red2"), 
                       name = "P-stwo\ndefaultu") +
  
  # WARSTWA 2: Punkty
  geom_jitter(data = test_baked, 
              aes(x = PAY_0, y = BILL_AMT1, color = as.factor(default)), 
              alpha = 0.8, size = 1.8, width = 0.2) + 
  
  scale_color_manual(values = c("0" = "darkgreen", "1" = "darkred"), 
                     name = "Rzeczywisty\nstatus (Test)",
                     labels = c("Spłata", "Default")) +
  
  scale_x_continuous(
    breaks = seq(-2, 8, by = 1), 
    limits = c(-2.5, 8.5),
    expand = c(0, 0)
  ) +
  
  theme_minimal() +
  labs(title = "Weryfikacja modelu: Interakcja PAY_0 i BILL_AMT1",
       subtitle = "Mapa prawdopodobieństwa (model) vs rzeczywiste dane (zbiór testowy)",
       x = "Status płatności (PAY_0)",
       y = "Kwota rachunku (BILL_AMT1)") +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

# -------------------------------------------------------------------------
# ETAP 6: Analiza siły interakcji (H-statistic)
# -------------------------------------------------------------------------

top_4_vars <- vi(rf_obj) %>% 
  top_n(4, wt = Importance) %>% 
  pull(Variable)

# 2. Automatyczne pary
pary <- split(combn(top_4_vars, 2), col(combn(top_4_vars, 2)))

# 3. Obliczenia i sumowanie (pętla)
interact_list <- lapply(pary, function(p) {
  res <- vi_firm(rf_obj, feature_names = p, train = train_baked, interaction = TRUE)
  data.frame(
    Para = paste(p, collapse = " & "),
    Suma_H_stat = sum(res$Importance)
  )
})

# 4. Finalna tabela i wykres
interact_df <- bind_rows(interact_list)

ggplot(interact_df, aes(x = Suma_H_stat, y = reorder(Para, Suma_H_stat))) +
  geom_col(fill = "#1a5276", width = 0.6) +
  geom_text(aes(label = round(Suma_H_stat, 3)), 
            hjust = -0.2, 
            fontface = "bold", 
            size = 3.5) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) + 
  theme_minimal() +
  labs(title = "Siła interakcji między kluczowymi zmiennymi",
       subtitle = "Automatyczna analiza kombinacji Top 4 predyktorów",
       x = "Suma H-stat", 
       y = "Para zmiennych") +
  theme(
    axis.text.y = element_text(size = 9),
    plot.title = element_text(face = "bold"),
  )


# -------------------------------------------------------------------------
# ZAPIS MODELU FINALNEGO
# -------------------------------------------------------------------------
  
if (!dir.exists("models")) {
  dir.create("models")
}

saveRDS(final_rf_fit, "models/final_model.rds")
cat("\nModel finalny (Random Forest) został zapisany w pliku: final_model.rds\n")