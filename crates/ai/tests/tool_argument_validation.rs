//! Public contract for validating parsed tool-call arguments before execution.

use banshu_ai::{Tool, ToolValidationError};
use serde_json::{Value, json};

fn itinerary_tool() -> Tool {
    Tool {
        name: "plan_itinerary".into(),
        description: "Plan a trip with constrained stops".into(),
        parameters: json!({
            "type": "object",
            "required": ["mode", "traveler", "stops"],
            "properties": {
                "mode": { "const": "rail" },
                "traveler": {
                    "type": "object",
                    "required": ["name", "age"],
                    "properties": {
                        "name": { "type": "string", "minLength": 2, "maxLength": 40 },
                        "age": { "type": "integer", "minimum": 18, "maximum": 120 }
                    },
                    "additionalProperties": false
                },
                "stops": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["city", "nights"],
                        "properties": {
                            "city": { "type": "string", "enum": ["Paris", "Lyon"] },
                            "nights": { "type": "number", "exclusiveMinimum": 0, "multipleOf": 0.5 }
                        },
                        "additionalProperties": false
                    }
                }
            },
            "additionalProperties": false
        }),
        strict: true,
    }
}

fn valid_itinerary_arguments() -> Value {
    json!({
        "mode": "rail",
        "traveler": { "name": "Ada", "age": 36 },
        "stops": [{ "city": "Paris", "nights": 2 }]
    })
}

fn arguments_with(path: &str, invalid: Value) -> Value {
    let mut arguments = valid_itinerary_arguments();
    *arguments.pointer_mut(path).unwrap() = invalid;
    arguments
}

#[test]
fn nested_valid_arguments_are_returned_unchanged() {
    let tool = itinerary_tool();
    let arguments = json!({
        "mode": "rail",
        "traveler": { "name": "Ada", "age": 36 },
        "stops": [
            { "city": "Paris", "nights": 2.5 },
            { "city": "Lyon", "nights": 1 }
        ]
    });
    let before = arguments.clone();

    let validated: Value = tool
        .validate_arguments(&arguments)
        .expect("nested arguments should satisfy the tool schema");

    assert_eq!(validated, before);
    assert_eq!(arguments, before, "validation must not mutate caller input");
}

#[test]
fn primitive_type_failure_is_structured_without_coercion_or_mutation() {
    let tool = itinerary_tool();
    let cases = [
        ("string to integer", json!("36"), "/traveler/age", "integer"),
        ("number to string", json!(36), "/traveler/name", "string"),
        ("boolean to string", json!(true), "/traveler/name", "string"),
    ];

    for (label, invalid, path, expected_type) in cases {
        let arguments = arguments_with(path, invalid);
        let before = arguments.clone();

        let error: ToolValidationError = tool
            .validate_arguments(&arguments)
            .err()
            .unwrap_or_else(|| panic!("{label} must not be coerced"));

        assert_eq!(error.tool_name, "plan_itinerary");
        assert_eq!(error.path, path);
        assert_eq!(error.constraint, "type");
        assert!(
            error.reason.contains(expected_type),
            "{label}: {}",
            error.reason
        );
        assert_eq!(arguments, before, "{label} must not mutate caller input");
    }
}

#[test]
fn missing_nested_required_property_reports_the_containing_object() {
    let tool = itinerary_tool();
    let arguments = json!({
        "mode": "rail",
        "traveler": { "name": "Ada" },
        "stops": [{ "city": "Paris", "nights": 2 }]
    });

    let error = tool.validate_arguments(&arguments).unwrap_err();

    assert_eq!(error.path, "/traveler");
    assert_eq!(error.constraint, "required");
    assert!(error.reason.contains("age"));
}

#[test]
fn supported_schema_constraints_reject_representative_failures() {
    let tool = itinerary_tool();
    let mut additional_property = valid_itinerary_arguments();
    additional_property["traveler"]["vip"] = json!(true);
    let cases = [
        (json!([]), "", "type"),
        (arguments_with("/stops", json!("Paris")), "/stops", "type"),
        (additional_property, "/traveler", "additionalProperties"),
        (arguments_with("/mode", json!("air")), "/mode", "const"),
        (
            arguments_with("/stops/0/city", json!("Berlin")),
            "/stops/0/city",
            "enum",
        ),
        (
            arguments_with("/traveler/name", json!("A")),
            "/traveler/name",
            "minLength",
        ),
        (
            arguments_with("/traveler/age", json!(17)),
            "/traveler/age",
            "minimum",
        ),
        (arguments_with("/stops", json!([])), "/stops", "minItems"),
        (
            arguments_with("/stops/0/nights", json!(0)),
            "/stops/0/nights",
            "exclusiveMinimum",
        ),
        (
            arguments_with("/stops/0/nights", json!(1.2)),
            "/stops/0/nights",
            "multipleOf",
        ),
    ];

    for (arguments, expected_path, expected_constraint) in cases {
        let error = tool.validate_arguments(&arguments).unwrap_err();
        assert_eq!(error.path, expected_path, "arguments: {arguments}");
        assert_eq!(
            error.constraint, expected_constraint,
            "arguments: {arguments}"
        );
        assert!(!error.reason.is_empty());
    }
}
