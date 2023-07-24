# Changelog

All notable changes to this project will be documented in this file.

## [0.7.1] - 2023-07-23

## Added

- Added Normalizer.

### Updated

- User Interface.
- Knowledge Base with new information.

## [0.7.0] - 2023-07-9

### Added

- Added GPT-4 and GPT-4-Prompt-Engineering to the models.

## [0.6.0] - 2023-06-21

### Added

- Added the block of Post-Process.
- Added the fine-tuned Ada model for classification.
- Added the Classifier to Post-Process block.
- Added the model with the Classifier setting:
    * GPT-Turbo-Prompt-Engineering-Responses-9-Non-ChitChat


## [0.5.1] - 2023-06-5

### Fix

- Bug in chat completion requests.

### Changed

- Temperature for models.

### Removed

- GPT-4 due to API unavailability.

## [0.5.0] - 2023-05-22

### Added

- Added vector database (Faiss)
- Added ZIBERT_v2 vectorizer model
- Added OpenAI ADA vectorizer model
- Added search context within the vector database
- Added auto config generator for different models
- Added support for storing history of the conversation and feeding it to model.
- Added message convertor to text format.
- Added support for additional models (with context injection):
    * GPT-Turbo-Prompt-Engineering
    * GPT-4-Prompt-Engineering
    * GPT-Davinci-Prompt-Engineering
    * GPT-Davinci-9Epoch-Prompt-Engineering

### Changed

- Refactored the code
- Changed the default message handler to chat format.

## [0.4.0] - 2023-05-14

### Changed

- Changed the config for prompt creation in chat models.

### Added

- Added support of additional models:
    * Finetuned GPT davinci on FAQ 9 Epochs (davinci:ft-personal:faq-9epoch-2023-05-13-17-23-45)
    * Finetuned GPT davinci on Chat dataset (davinci:ft-personal:chat-2023-05-13-20-08-50)

## [0.3.0] - 2023-05-08

### Added

- Added support of additional models:
    * Finetuned GPT davinci on FAQ (davinci:ft-personal:faq-2023-05-07-05-25-57)

## [0.2.0] - 2023-05-01

### Added

- Added support of additional models:
    * GPT-4 (gpt-4)
    * ChatGPT (gpt-3.5-turbo)
    * GPT-3.5 Davinci (text-davinci-003)
    * Finetuned GPT davinci on RBT Questions 
    (davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56)
- Added default questions with answers from database for finetune models.

## [0.1.0] - 2023-04-22

### Added

- Added inference session for 'gpt3.5-turbo' model.
- UI interface.
