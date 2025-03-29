# API Configuration Guide

This guide explains how to configure the various API integrations in BioData Explorer to access external biological databases and AI services.

## Overview of API Integrations

BioData Explorer integrates with several external services:

1. **NCBI Databases**: For accessing nucleotide, protein, gene, and publication data
2. **UniProt API**: For accessing detailed protein information
3. **OpenAI API**: For AI-powered text analysis and biological insights
4. **Protein Language Models**: For advanced protein sequence analysis

Each integration requires specific configuration to function properly.

## NCBI API Configuration

### Basic Configuration

NCBI's E-utilities API requires:
- User email address (mandatory)
- API key (optional, but recommended for higher request limits)

### Setup Steps

1. Navigate to "Genomic API" → "API Configuration" in the sidebar
2. In the "NCBI Configuration" section:
   - Enter your email address in the "Email for NCBI API" field
   - Optionally, enter your NCBI API key if you have one

### Obtaining an NCBI API Key

To get an NCBI API key (recommended for researchers):

1. Create an NCBI account at [https://www.ncbi.nlm.nih.gov/](https://www.ncbi.nlm.nih.gov/)
2. Log in to your NCBI account
3. Go to your account settings
4. Navigate to "API Key Management"
5. Generate a new API key
6. Copy the key and paste it into the BioData Explorer configuration

### Usage Limits

Without an API key:
- Limited to 3 requests per second
- May face IP-based throttling for intensive use

With an API key:
- Up to 10 requests per second
- More reliable access during peak times
- Higher priority for resource-intensive queries

## UniProt API Configuration

### Basic Information

The UniProt API:
- Does not require authentication for basic access
- Has rate limits for unauthenticated users
- Provides comprehensive protein data

### Configuration (If Needed)

For specialized or high-volume access:

1. Navigate to "Genomic API" → "API Configuration" in the sidebar
2. In the "UniProt Configuration" section:
   - Enable "Use Advanced UniProt Access" if needed
   - Enter any access credentials if you have a special arrangement with UniProt

### Usage Considerations

- Respect the [UniProt usage guidelines](https://www.uniprot.org/help/about)
- For bulk downloads or intensive querying, consider downloading datasets directly from UniProt

## OpenAI API Configuration

### Requirements

OpenAI integration requires:
- An OpenAI API key (mandatory)
- Selection of AI model to use

### Setup Steps

1. Navigate to "AI Analysis" → "API Configuration" in the sidebar
2. In the "OpenAI Configuration" section:
   - Enter your OpenAI API key
   - Select which model to use by default (e.g., GPT-3.5-turbo, GPT-4)
   - Configure maximum token settings if needed

### Obtaining an OpenAI API Key

To get an OpenAI API key:

1. Create an account at [https://openai.com/](https://openai.com/)
2. Log in to your account
3. Navigate to "API keys" in your account settings
4. Create a new API key
5. Copy the key and paste it into the BioData Explorer configuration

### Cost Considerations

Be aware that using OpenAI's API incurs costs:
- Different models have different pricing
- Token usage affects costs
- BioData Explorer shows estimated token usage before processing

### Model Selection Guide

- **GPT-3.5-turbo**: Faster, lower cost, good for most analyses
- **GPT-4**: Higher quality, better for complex biological analyses, higher cost
- **GPT-4-32k**: Extended context window for analyzing long research papers, highest cost

## Protein Language Model Configuration

### Local vs. Remote Inference

BioData Explorer supports:
- Local inference (using downloaded models)
- Remote inference (using API-based services)

### Local Model Setup

For local protein language model inference:

1. Navigate to "AI Analysis" → "Model Configuration" in the sidebar
2. In the "Protein Model Configuration" section:
   - Select "Use Local Models"
   - Choose model size based on your computing resources
   - Configure cache settings if needed

### Remote API Configuration

For remote protein language model inference:

1. Navigate to "AI Analysis" → "Model Configuration" in the sidebar
2. In the "Protein Model Configuration" section:
   - Select "Use Remote API"
   - Enter API endpoint information
   - Provide authentication credentials if required

### Model Selection Guide

- **ESM-2 (650M)**: Good balance of speed and accuracy
- **ESM-2 (3B)**: Higher accuracy, requires more computing resources
- **ESM-1b**: Older model, faster but less accurate

## Secure API Key Management

BioData Explorer provides secure API key management:

- Keys are stored only in your session
- Keys are never shared or saved between sessions
- For regular usage, you'll need to re-enter keys when starting a new session

### Best Practices

1. Never share your API keys with others
2. Regularly rotate your API keys for security
3. Set up usage limits in the API provider dashboards
4. Monitor API usage to control costs

## Troubleshooting API Connections

### Common NCBI API Issues

- **"Invalid email"**: Ensure you've entered a valid email address
- **"Too many requests"**: You're exceeding rate limits, consider getting an API key
- **"ID not found"**: Check that the sequence/gene ID exists and is formatted correctly

### Common OpenAI API Issues

- **"Authentication error"**: Verify your API key is correct and not expired
- **"Rate limit exceeded"**: You've hit usage limits, wait or adjust requests
- **"Token limit exceeded"**: Your input text is too long, consider chunking

### General Troubleshooting Steps

1. Verify API credentials are entered correctly
2. Check internet connectivity
3. Ensure the external service is operational
4. Look for error messages in the application logs
5. Try a simple query to test connectivity

## API Feature Availability

| Feature | Required API | Alternative |
|---------|--------------|-------------|
| NCBI Sequence Search | NCBI Email | None |
| Protein Information | None (UniProt) | NCBI API |
| Research Paper Analysis | OpenAI API | None |
| Protein Function Prediction | OpenAI API or Protein LM | Basic sequence analysis |

## Privacy and Data Handling

When using API integrations, be aware that:

1. Data sent to external APIs may be subject to their terms of service
2. OpenAI may store queries for service improvement
3. Consider data sensitivity when using external services

For sensitive research data, consider:
- Using local models where possible
- Anonymizing data before API submission
- Checking service provider's data handling policies

---

For further assistance, contact the developer: AlMotasem Bellah Younis, PhD