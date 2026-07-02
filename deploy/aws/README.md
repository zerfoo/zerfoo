# Zerfoo on AWS Marketplace

This directory contains everything needed to list and deploy Zerfoo on the
AWS Marketplace as a SaaS product.

## Files

| File | Purpose |
|------|---------|
| `cloudformation.yaml` | CloudFormation template — ECS Fargate, ALB, autoscaling, CloudWatch |
| `metering.go` | AWS Marketplace Metering API integration (interface-based, no SDK dep) |
| `metering_test.go` | Unit and integration tests for metering |
| `listing.json` | AWS Marketplace listing metadata (title, pricing, dimensions) |
| `Dockerfile` | Multi-stage container image for the marketplace listing |
| `README.md` | This file |

## Prerequisites

- AWS account with Marketplace Seller access
- AWS CLI v2 configured with appropriate credentials
- Docker (for building and pushing the container image)
- An ECR repository in your AWS account

## Pricing Model

Zerfoo uses the **SaaS** pricing model (not AMI). Customers subscribe through
the Marketplace and usage is metered along three dimensions:

| Dimension | Unit | Description |
|-----------|------|-------------|
| `inference-requests` | requests | Each call to `/v1/chat/completions` or `/v1/completions` |
| `tokens-processed` | tokens | Sum of input + output token counts per request |
| `gpu-hours` | hours | GPU compute time consumed (reported in whole hours) |

## Publishing Steps

### 1. Build and Push the Container Image

```bash
# Authenticate to ECR Marketplace registry (us-east-1 only for listing).
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin public.ecr.aws

# Build the image.
docker build -t zerfoo:latest -f deploy/aws/Dockerfile .

# Tag and push to your ECR repository.
ECR_URI=709825985650.dkr.ecr.us-east-1.amazonaws.com/zerfoo/zerfoo
docker tag zerfoo:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

### 2. Create the Marketplace Listing

1. Log in to [AWS Marketplace Management Portal](https://aws.amazon.com/marketplace/management/).
2. Choose **Products → Server** then **Create product → SaaS**.
3. Fill in the product details from `listing.json` (title, descriptions, highlights).
4. Upload the logo from the URL in `listing.json`.
5. Configure the three metering dimensions defined in `listing.json`.
6. Submit for AWS review (typically 3–5 business days).
7. After approval, note the **Product Code** assigned by AWS.

### 3. Update the Product Code

Replace `PLACEHOLDER_PRODUCT_CODE` in `listing.json` and set it as an
environment variable in the CloudFormation stack:

```bash
PRODUCT_CODE=<your-aws-marketplace-product-code>
```

### 4. Deploy with CloudFormation

```bash
aws cloudformation deploy \
  --template-file deploy/aws/cloudformation.yaml \
  --stack-name zerfoo-marketplace \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProductCode=${PRODUCT_CODE} \
    ContainerImage=${ECR_URI}:latest \
    VpcId=<your-vpc-id> \
    SubnetIds=<subnet-1,subnet-2> \
    CertificateArn=<acm-cert-arn>
```

The stack creates:
- An ECS Fargate cluster and service
- An Application Load Balancer (HTTP → HTTPS redirect when cert provided)
- Target tracking autoscaling (default: 1–10 tasks, target 50 req/task)
- CloudWatch log group and alarms (P99 latency, unhealthy tasks, 5xx rate)
- IAM roles with least-privilege Marketplace metering permissions

### 5. Verify the Deployment

```bash
# Get the ALB DNS name.
aws cloudformation describe-stacks \
  --stack-name zerfoo-marketplace \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
  --output text

# Health check.
curl http://<alb-dns>/health

# Test inference (OpenAI-compatible).
curl http://<alb-dns>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello"}]}'
```

## Metering Integration

Metering is performed by the Zerfoo serve process via the `deploy/aws` package.
The `Meter` type wraps a `MeteringClient` interface so it can be swapped for
an AWS SDK client in production:

```go
import "github.com/zerfoo/zerfoo/deploy/aws"

// In production, replace HTTPMeteringClient with an SDK-backed implementation.
client := aws.NewHTTPMeteringClient(
    "https://metering.marketplace.us-east-1.amazonaws.com",
    os.Getenv("ZERFOO_MARKETPLACE_PRODUCT_CODE"),
)
meter := aws.NewMeter(client, productCode, customerID)

// Report usage after each request.
_ = meter.RecordInferenceRequests(ctx, 1)
_ = meter.RecordTokensProcessed(ctx, inputTokens+outputTokens)
```

## Autoscaling

The CloudFormation template uses ALB request-count-per-target autoscaling.
Tune `TargetRequestsPerTask`, `MinCapacity`, and `MaxCapacity` to match your
expected workload. Scale-out cooldown is 60 s; scale-in cooldown is 300 s.

## CloudWatch Alarms

Three alarms are created automatically:

| Alarm | Threshold | Action |
|-------|-----------|--------|
| `high-p99-latency` | P99 > 5 s for 3 min | Investigate task resources |
| `unhealthy-tasks` | Any unhealthy task for 2 min | Check ECS events and logs |
| `high-5xx` | > 10 errors/min for 3 min | Review application logs |

Connect these alarms to an SNS topic for PagerDuty / email notifications.

## Security Notes

- ECS tasks run in private subnets; only the ALB is internet-facing.
- The task IAM role is scoped to `aws-marketplace:MeterUsage` and
  `cloudwatch:PutMetricData` only.
- Enable VPC Flow Logs and AWS CloudTrail in your account for audit trails.
- Rotate ECR image tags on every release; avoid using `:latest` in production.
