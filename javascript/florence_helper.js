function send_florence_prompt(prompt_text, target_tab) {
    const textarea_id = target_tab + "_prompt";
    const textarea = document.getElementById(textarea_id);
    if (textarea) {
        const native_textarea = textarea.querySelector("textarea");
        if (native_textarea) {
            native_textarea.value = prompt_text;
            // Dispatch input event so Gradio/Svelte UI updates the bindings
            native_textarea.dispatchEvent(new Event("input", { bubbles: true }));
            
            // Search and click the target tab button to switch
            const tab_buttons = document.querySelectorAll("button");
            for (const btn of tab_buttons) {
                if (btn.innerText.trim().toLowerCase() === target_tab) {
                    btn.click();
                    break;
                }
            }
        }
    }
}
