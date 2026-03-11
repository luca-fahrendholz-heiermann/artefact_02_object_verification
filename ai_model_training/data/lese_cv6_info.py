import ijson

FN = r"cv6_info.json"

with open(FN, "rb") as f:
    for i, obj in enumerate(ijson.items(f, "fold0.train.item"), 1):
        # obj ist ein Dict wie {"esf_ref":..., "esf_scan":..., "label":...}
        if obj.get("label") ==2:
            print(f"#{i}")
            #print("  esf_ref :", obj.get("esf_ref"))
            print("  esf_scan:", obj.get("esf_scan"))
            #print("  label   :", obj.get("label"))