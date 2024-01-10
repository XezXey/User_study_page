function generateEvaluationTable(sj_name, relightingType, orders, imgPath) {
    console.log("GGGA", "generateEvaluationTable", sj_name, relightingType, orders, imgPath);
    let output_html = `
    <table class="face-table" style="margin-left: auto;margin-right: auto;">
    <thead>
    </thead>
    `;

    for (let i = 0; i < orders.length; i++) {
        const method_name = orders[i];
        const img_tmp_path = `${imgPath}/${sj_name}/${method_name}_${sj_name}.png`;

        if (method_name !== "input") {
            output_html += `<td><div class="mint"><span style="font-size: 20px;">${relightingType}#${i}</span><br><img src="${img_tmp_path}" class="face-img"/></div></td>`;
        }
    }

    var questions = ["Most Accurate", "Most Realistic"]; // Add more questions as needed
    for (let q = 0; q < questions.length; q++) {
        output_html += `<tr><td colspan=${orders.length} style="text-align: center;">${questions[q]}:`;
        for (let i = 0; i < orders.length; i++) {
            const method_name = orders[i];
            if (method_name !== "input") {
                output_html += `<input type="radio" id="${method_name}_${questions[q]}_${sj_name}" name="${questions[q]}_${sj_name}" value="${method_name}"><label for="${method_name}_${questions[q]}_${sj_name}">${relightingType}#${i}</label>`;
            }
        }
        output_html += `</td></tr>`;
    }

    output_html += `</tr></table>`;
    return output_html;
}
