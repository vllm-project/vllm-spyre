# Other Tests

!!! note
    Unless otherwise specified, all the continuous batching tests are running with `max_model_len=512`

::: tests.e2e.test_spyre_cb
    options:
        show_root_heading: true

::: tests.e2e.test_spyre_async_llm
    options:
        show_root_heading: true
        members:
        - test_abort

::: tests.e2e.test_spyre_max_new_tokens
    options:
        show_root_heading: true
        members:
        - test_output

::: tests.e2e.test_spyre_online
    options:
        show_root_heading: true
        members:
        - test_openai_serving_cb
